from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr


class UserCreate(BaseModel):
    """Schema for user signup/registration."""
    email: EmailStr = Field(..., description="User's email address")
    username: str = Field(..., min_length=3, max_length=100, description="Display name")
    password: str = Field(..., min_length=8, max_length=128, description="Plaintext password")
    role: Optional[str] = Field(default="student", max_length=50, description="User role")


class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., description="Plaintext password")


class UserResponse(BaseModel):
    """Schema for user profile response."""
    uid: str
    email: EmailStr
    username: str
    role: str
    is_verified: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    """Schema for JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class ForgotPassword(BaseModel):
    """Schema for password reset request."""
    email: EmailStr = Field(..., description="User's email address")


class ResetPassword(BaseModel):
    """Schema for new password submission."""
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")
    confirm_password: str = Field(..., min_length=8, max_length=128, description="Password confirmation")


class UserUpdate(BaseModel):
    """Schema for updating user profile."""
    username: Optional[str] = Field(default=None, max_length=100)
    role: Optional[str] = Field(default=None, max_length=50)