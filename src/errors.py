from fastapi import HTTPException, status


class InvalidToken(HTTPException):
    def __init__(self, detail: str = "Invalid token"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


class RevokedToken(HTTPException):
    def __init__(self, detail: str = "Token has been revoked"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


class AccessTokenRequired(HTTPException):
    def __init__(self, detail: str = "Access token required"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


class RefreshTokenRequired(HTTPException):
    def __init__(self, detail: str = "Refresh token required"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


class AccountNotVerified(HTTPException):
    def __init__(self, detail: str = "Account not verified. Please check your email."):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


class InsufficientPermission(HTTPException):
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


class InvalidCredentials(HTTPException):
    def __init__(self, detail: str = "Invalid email or password"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


class UserNotFound(HTTPException):
    def __init__(self, detail: str = "User not found"):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


class UserAlreadyExists(HTTPException):
    def __init__(self, detail: str = "User with this email already exists"):
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=detail)