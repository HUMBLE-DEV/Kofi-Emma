from dotenv import load_dotenv
load_dotenv()


from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import EmailStr, Field
import os


class Settings(BaseSettings):
    PROJECT_NAME: str = "ScholarGraph"
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Public base URL used to build email verification / password reset links
    PUBLIC_BASE_URL: str

    # Frontend base URL (Streamlit app) for password reset links
    FRONTEND_BASE_URL: str = ""

    # When FRONTEND_BASE_URL is empty, PUBLIC_BASE_URL is used for both backend and frontend

    # Database configuration strings
    DATABASE_URL: str='sqlite+aiosqlite:///./research_agent.db'
    REDIS_URL: str 
    #QDRANT_URL: str

    # Third-Party Integrations
    GROQ_API_KEY: str
    TAVILY_API_KEY: str

    # Mail service
    MAIL_USERNAME: str
    MAIL_PASSWORD: str
    MAIL_FROM: EmailStr
    MAIL_PORT: int = 587
    MAIL_SERVER: str = "smtp.gmail.com"
    MAIL_STARTTLS: bool = True
    MAIL_SSL_TLS: bool = False
    MAIL_FROM_NAME: str = "ScholarGraph"

    # When true, a Redis outage fails auth requests closed instead of open.
    TOKEN_BLOCKLIST_STRICT: bool = False

    # Comma-separated list of allowed CORS origins for the FastAPI backend.
    CORS_ORIGINS: str = "http://localhost:8501,https://scholargraph.duckdns.org"
    ALLOWED_HOSTS: str = "*"

    DEFAULT_MODEL: str = "gemini-2.5-flash"
    GEMINI_MODELS: str = "gemini-2.5-flash,gemini-3.5-flash-lite"
    FALLBACK_GROQ_MODEL: str = "openai/gpt-oss-20b"
    
    MAX_CONTEXT_TOKENS: int = 3500

    # ---- Storage -------------------------------------------------------------
    UPLOAD_DIR: str = "uploads"
    CHROMA_DIR: str = "chroma_db"
    CHROMA_HOST: str = ""  # set to run Chroma as its own server (recommended with Celery)
    CHROMA_PORT: int = 8000
    SEMANTIC_SCHOLAR_API_KEY: str = ""

    MAX_MESSAGE_CHARS:int=4000
    MAX_UPLOAD_MB:int=20
    MAX_DOCS_PER_THREAD:int=4
    CHAT_RATE_LIMIT_PER_MIN:int=20
    UPLOAD_RATE_LIMIT_PER_HOUR:int=4
    GRAPH_RECURSION_LIMIT:int=16
    MAX_CONTEXT_TOKENS: int = 8000
    MAX_TOOL_CALLS_PER_TURN:int=2
    TURN_TIMEOUT_SECONDS:int=180
    SAVE_PAPER_RATE_LIMIT_PER_HOUR:int=30
    PAPER_SEARCH_RATE_LIMIT_PER_MIN:int=5


    model_config=SettingsConfigDict(
            env_file=".env",
            extra="ignore"
        )
settings = Settings()