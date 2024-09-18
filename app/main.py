from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware import RequestLoggingMiddleware, TimingMiddleware
from app.api.routes import documents, health, query
from app.config import settings
from app.database import init_db

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if settings.log_level == "DEBUG" else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        structlog.get_level_from_name(settings.log_level)
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("startup", message="Initializing database and pgvector extension")
    try:
        await init_db()
        logger.info("startup", message="Database initialized successfully")
    except Exception as e:
        logger.error("startup_failed", error=str(e))
    yield
    logger.info("shutdown", message="Application shutting down")


app = FastAPI(
    title="DocMind",
    description="Document Q&A platform with Retrieval-Augmented Generation. Upload documents, ask questions, get cited answers.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(TimingMiddleware)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)
