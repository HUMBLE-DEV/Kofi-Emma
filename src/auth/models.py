import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel

import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlmodel import SQLModel, Field, Relationship


class User(SQLModel, table=True):
    __tablename__ = "user"

    uid: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    email: str = Field(index=True, unique=True, max_length=255)
    username: str = Field(index=True, unique=True, max_length=100)
    password_hash: str = Field(exclude=True)
    role: str = Field(default="student", max_length=50)
    is_verified: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # RELATIONSHIP: Links user down to their individual conversations
    conversations: List["Conversation"] = Relationship(back_populates="user", cascade_delete=True)


class Conversation(SQLModel, table=True): 
    __tablename__ = "conversations"

    uid: Optional[int] = Field(default=None, primary_key=True, index=True)
    thread_id: str = Field(index=True, unique=True)
    title: str = Field(default="New Chat")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # FOREIGN KEY: Links this conversation explicitly to a single unique User account
    user_uid: uuid.UUID = Field(foreign_key="user.uid", index=True, ondelete="CASCADE")
    user: User = Relationship(back_populates="conversations")

    # RELATIONSHIPS: Links down to messages and long term contextual summaries
    messages: List["ChatMessage"] = Relationship(back_populates="conversation", cascade_delete=True)
    memories: List["LongTermMemory"] = Relationship(back_populates="conversation", cascade_delete=True)


class ChatMessage(SQLModel, table=True): 
    __tablename__ = "chat_messages"

    uid: Optional[int] = Field(default=None, primary_key=True, index=True)
    role: str = Field(max_length=50)
    content: str = Field()
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # FOREIGN KEY: Links message back to its containing parent conversation thread
    conversation_thread_id: str = Field(foreign_key="conversations.thread_id", index=True, ondelete="CASCADE")
    conversation: Conversation = Relationship(back_populates="messages")


class LongTermMemory(SQLModel, table=True): 
    __tablename__ = "long_term_memory"

    uid: Optional[int] = Field(default=None, primary_key=True, index=True)
    memory: str = Field()
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # FOREIGN KEY: Links memory block to a specific context thread execution chain
    conversation_thread_id: str = Field(foreign_key="conversations.thread_id", index=True, ondelete="CASCADE")
    conversation: Conversation = Relationship(back_populates="memories")
