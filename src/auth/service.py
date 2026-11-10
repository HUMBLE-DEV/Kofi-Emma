import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.models import User
from src.auth.schema import UserCreate
from src.auth.utility import hash_password
from src.errors import UserAlreadyExists


class AuthService:
    """Business logic layer for authentication and account management."""

    async def create_user(self, user_data: UserCreate, session: AsyncSession) -> User:
        """Register a new user with a hashed password and an unverified status."""
        existing = await self.get_user_by_email(user_data.email, session)
        if existing:
            raise UserAlreadyExists()

        existing_name = await self.get_user_by_username(user_data.username, session)
        if existing_name:
            raise UserAlreadyExists(detail="That username is already taken")

        new_user = User(
            uid=uuid.uuid4(),
            email=user_data.email.lower(),
            username=user_data.username,
            password_hash=hash_password(user_data.password),
            role=user_data.role or "student",
            is_verified=False,
        )

        session.add(new_user)
        try:
            await session.commit()
        except IntegrityError:
            # Lost a race against a concurrent signup with the same email/username.
            await session.rollback()
            raise UserAlreadyExists()
        await session.refresh(new_user)
        return new_user

    async def get_user_by_username(self, username: str, session: AsyncSession) -> Optional[User]:
        """Look up a user by their display name."""
        statement = select(User).where(User.username == username)
        result = await session.execute(statement)
        return result.scalar_one_or_none()

    async def get_user_by_email(self, email: str, session: AsyncSession) -> Optional[User]:
        """Look up a user by email address (case-insensitive)."""
        statement = select(User).where(User.email == email.lower())
        result = await session.execute(statement)
        return result.scalar_one_or_none()

    async def get_user_by_uid(self, uid, session: AsyncSession) -> Optional[User]:
        """Look up a user by their unique identifier."""
        statement = select(User).where(User.uid == uid)
        result = await session.execute(statement)
        return result.scalar_one_or_none()

    async def update_user(
        self,
        user: User,
        attributes: dict,
        session: AsyncSession,
    ) -> User:
        """Apply a set of attribute changes to a user and persist them."""
        for key, value in attributes.items():
            setattr(user, key, value)

        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user

    async def set_password(self, user: User, new_password: str, session: AsyncSession) -> User:
        """Replace a user's password with a freshly hashed version."""
        return await self.update_user(user, {"password_hash": hash_password(new_password)}, session)