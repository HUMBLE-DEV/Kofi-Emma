from urllib.parse import quote
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlmodel.ext.asyncio.session import AsyncSession
from datetime import timedelta
from backend.guardrails import rate_limit
from src.db.main import get_session
from src.auth.service import AuthService
from src.auth.models import User
from src.auth.schema import (
    UserCreate,
    UserLogin,
    UserResponse,
    ForgotPassword,
    ResetPassword,
)
from src.auth.utility import (
    verify_password,
    create_access_token,
    create_url_safe_token,
    decode_url_safe_token,
)
from src.auth.dependency import (
    RoleChecker,
    get_current_user,
    AccessTokenBearer,
    RefreshTokenBearer,
    add_jti_to_blocklist,
)
from src.mail import mail as fast_mail, create_message
from src.celery_worker import Send_email
from src.errors import InvalidCredentials, UserNotFound
from src.config import settings
import logging

logger = logging.getLogger("ScholarGraph-auth")

auth_router = APIRouter()
auth_service = AuthService()
admin_required = RoleChecker(["admin"])

def queue_email(recipients: list, subject: str, body: str) -> None:
    """Enqueue a verification / password-reset email.

    Runs in a threadpool via FastAPI's BackgroundTasks so the HTTP response
    returns immediately and the broker/SMTP latency never blocks the client.
    Falls back to direct SMTP delivery if Celery is unavailable.
    """
    try:
        Send_email.delay(recipients=recipients, subject=subject, body=body)
    except Exception as exc:
        logger.warning("Celery unavailable (%s); sending the email directly.", exc)
        try:
            import asyncio
            asyncio.run(fast_mail.send_message(create_message(recipients, subject, body)))
        except Exception as exc2:
            logger.error("Direct email delivery also failed: %s", exc2)


@auth_router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(
    user_data: UserCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    new_user = await auth_service.create_user(user_data, session)
    token = create_url_safe_token({"email": new_user.email})
    link = f"{settings.PUBLIC_BASE_URL}/api/v1/auth/verify?token={token}"
    html = f"""
        <h2>Welcome to ScholarGraph!</h2>
        <p>Hello {new_user.username},</p>
        <p>Please verify your email to begin your research journey.</p>
        <p><a href="{link}">Click here to verify your account</a></p>
        <p>This link expires in 1 hour.</p>
    """
    background_tasks.add_task(
        queue_email,
        [new_user.email],
        "Verify your ScholarGraph account",
        html,
    )

    return {
        "message": "Account created. Check your email to verify.",
        "user": {
            "uid": str(new_user.uid),
            "email": new_user.email,
            "username": new_user.username,
            "role": new_user.role,
        },
    }


@auth_router.get("/verify")
async def verify_account(
    token: str,
    session: AsyncSession = Depends(get_session),
):
    token_data = decode_url_safe_token(token)
    if not token_data:
        return JSONResponse(
            content={"message": "Invalid or expired verification link"},
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    email = token_data.get("email")
    user = await auth_service.get_user_by_email(email, session)

    if not user:
        raise UserNotFound()

    if user.is_verified:
        return JSONResponse(
            content={"message": "Account already verified. You can login now."}
        )

    await auth_service.update_user(user, {"is_verified": True}, session)

    return JSONResponse(
        content={"message": "Account verified successfully. You can now login."}
    )


@auth_router.post("/login")
async def login(
    request: Request,
    user_data: UserLogin,
    session: AsyncSession = Depends(get_session),
):
    client_ip = request.client.host

    await rate_limit(
        user_id=client_ip,       # We use the IP instead of user.uid
        bucket="login_attempt", 
        limit=5, 
        window_seconds=60
        )
    
    # 3. Optional: Also rate limit by the Email they are trying to use
    await rate_limit(
        user_id=user_data.email,
        bucket="login_email",
        limit=5,
        window_seconds=60
        )
    user = await auth_service.get_user_by_email(user_data.email, session)

    if not user or not verify_password(user_data.password, user.password_hash):
        raise InvalidCredentials()

    access_token = create_access_token(
        user_data={
            "email": user.email,
            "user_id": str(user.uid),
            "role": user.role,
            "name": user.username,
        }
    )
    refresh_token = create_access_token(
        user_data={
            "email": user.email,
            "user_id": str(user.uid),
            "role": user.role,
            "name": user.username,
        },
        refresh=True,
        expiry=timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )

    return JSONResponse(
        content={
            "message": "Login successful",
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "email": user.email,
                "username": user.username,
                "uid": str(user.uid),
                "role": user.role,
                "is_verified": user.is_verified,
            },
        }
    )


@auth_router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return UserResponse(
        uid=str(current_user.uid),
        email=current_user.email,
        username=current_user.username,
        role=current_user.role,
        is_verified=current_user.is_verified,
        created_at=current_user.created_at,
    )


@auth_router.get("/refresh")
async def refresh_token(token_data: dict = Depends(RefreshTokenBearer())):
    access_token = create_access_token(token_data["user"])
    return JSONResponse(content={"access_token": access_token})


@auth_router.get("/logout")
async def logout(token_data: dict = Depends(AccessTokenBearer())):
    await add_jti_to_blocklist(token_data["jti"])
    return JSONResponse(content={"message": "Logged out successfully"})


@auth_router.post("/forgot_password")
async def forgot_password(
    data: ForgotPassword,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    user = await auth_service.get_user_by_email(data.email, session)

    if user is not None:
        token = create_url_safe_token({"email": user.email})
        frontend_url = settings.FRONTEND_BASE_URL or settings.PUBLIC_BASE_URL
        link = f"{frontend_url}?token={quote(token, safe='')}"
        html = f"""
            <h2>ScholarGraph - Password Reset</h2>
            <p>You requested a password reset for your account.</p>
            <p><a href="{link}">Click here to reset your password</a></p>
            <p>This link expires in 1 hour.</p>
            <p>If you did not request this, ignore this email.</p>
        """
        background_tasks.add_task(queue_email, [user.email], "Reset Password", html)

    return JSONResponse(
        content={
            "message": "If an account with this email exists you will receive a reset link shortly"
        },
        status_code=status.HTTP_200_OK,
    )


@auth_router.get("/reset_password")
async def reset_password_token_validation(
    token: str,
):
    """Validate a password reset token.

    Returns JSON indicating whether the token is valid and the associated email.
    """
    token_data = decode_url_safe_token(token)
    if not token_data:
        return JSONResponse(
            content={
                "valid": False,
                "reason": "Invalid or expired reset link. Please request a new one."
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    email = token_data.get("email")
    return JSONResponse(
        content={
            "valid": True,
            "email": email,
            "reason": None
        },
        status_code=status.HTTP_200_OK,
    )


@auth_router.post("/reset_password")
async def reset_password(
    password: ResetPassword,
    token: str,
    session: AsyncSession = Depends(get_session),
):
    if password.new_password != password.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match",
        )

    token_data = decode_url_safe_token(token)
    if not token_data:
        return JSONResponse(
            content={"message": "Invalid or expired reset link. Please request a new one."},
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    email = token_data.get("email")
    user = await auth_service.get_user_by_email(email, session)
    if not user:
        raise UserNotFound()

    await auth_service.set_password(user, password.new_password, session)

    return JSONResponse(
        content={"message": "Password reset successfully. You can now login."},
        status_code=status.HTTP_200_OK,
    )



@auth_router.delete("/{user_id}")
async def delete_user(
    user_id: uuid.UUID,
    _= Depends(admin_required),
    session: AsyncSession = Depends(get_session),
):
    from src.auth.models import User
    user = await auth_service.get_user_by_uid(user_id, session)
    if not user:
        raise UserNotFound()
    await session.delete(user)
    await session.commit()
    return {"message": "User deleted successfully"}


@auth_router.post("/resend-verification")
async def resend_verification(
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks = None,
):
    if current_user.is_verified:
        return JSONResponse(
            content={"message": "Account is already verified."},
            status_code=status.HTTP_200_OK,
        )

    token = create_url_safe_token({"email": current_user.email})
    link = f"{settings.PUBLIC_BASE_URL}/api/v1/auth/verify?token={token}"
    html = f"""
        <h2>Welcome to ScholarGraph!</h2>
        <p>Hello {current_user.username},</p>
        <p>Please verify your email to begin your research journey.</p>
        <p><a href="{link}">Click here to verify your account</a></p>
        <p>This link expires in 1 hour.</p>
    """
    queue_email(
        [current_user.email],
        "Verify your ScholarGraph account",
        html,
    )
    return JSONResponse(
        content={"message": "Verification email resent."},
        status_code=status.HTTP_200_OK,
    )