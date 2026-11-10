import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from src.config import settings
from src.logging_config import setup_app_logging

logger = setup_app_logging(name="ScholarGraph-middleware")


def register_middleware(app: FastAPI):
    """Attach CORS, host checking and request logging to the application."""

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            origin.strip()
            for origin in settings.CORS_ORIGINS.split(",")
            if origin.strip()
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[settings.ALLOWED_HOSTS],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = uuid.uuid4().hex[:8]
        start = time.time()
        try:
            response = await call_next(request)

        except Exception:
            logger.exception("[%s] %s %s crashed", request_id, request.method, request.url.path)
            raise
        response.headers["X-Request-ID"] = request_id
        duration = time.time() - start
        logger.info("[%s] %s  %s | %s | %s ",
         request_id, request.method,request.url.path,  response.status_code, duration
                    )
        return response

    

