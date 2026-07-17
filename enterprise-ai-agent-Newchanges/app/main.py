import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import logging
import secrets
import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app
from app.config import settings
from app.logger_config import configure_logging
from app.api.v1.routes import router as api_router, whisper_service
from app.db.redis_client import redis_client
from app.services.model_service import ModelService
from app.services.rag_service import RAGService
from app.services.sentiment_service import SentimentService
from app.services.voice_emotion_service import VoiceEmotionService


configure_logging()
logger = logging.getLogger(__name__)
Path("outputs").mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title=settings.app_name,
    description="Enterprise-grade AI assistant backend with RAG, memory and tool calling.",
    version="1.1.0",
)

cors_origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
allowed_hosts = [host.strip() for host in settings.allowed_hosts.split(",") if host.strip()] or ["*"]

app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=bool(cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def production_security(request: Request, call_next):
    """Apply request limits, optional API-key auth, correlation IDs and safe headers."""
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > settings.max_request_body_bytes:
                return JSONResponse(status_code=413, content={"detail": "request body is too large"})
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "invalid content-length header"})

    if settings.require_api_key and request.url.path.startswith("/api/") and request.url.path != "/api/v1/health":
        provided_key = request.headers.get("x-api-key", "")
        if not settings.api_key or not secrets.compare_digest(provided_key, settings.api_key):
            return JSONResponse(status_code=401, content={"detail": "invalid or missing API key"})

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

app.include_router(api_router, prefix="/api/v1")
app.mount("/metrics", make_asgi_app())
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.get("/")
async def chat_app():
    return FileResponse("app/static/index.html")


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Enterprise AI Assistant backend")
    if settings.require_api_key and not settings.api_key:
        raise RuntimeError("REQUIRE_API_KEY=true requires API_KEY to be configured")
    if settings.environment.lower() == "production" and not cors_origins:
        logger.warning("No CORS_ORIGINS configured; browser access is limited to same-origin requests.")
    try:
        await redis_client.connect()
    except Exception as e:
        logger.warning("Redis connection failed: %s. Continuing without Redis.", str(e))
    
    try:
        await ModelService.initialize()
    except Exception as e:
        logger.warning("Model initialization failed: %s. API will still start.", str(e))
    
    try:
        RAGService.initialize()
    except Exception as e:
        logger.warning("RAG service initialization failed: %s", str(e))

    try:
        SentimentService.initialize()
    except Exception as e:
        logger.warning("Sentiment service initialization failed: %s", str(e))

    try:
        VoiceEmotionService.initialize()
    except Exception as e:
        logger.warning("Voice emotion service initialization failed: %s", str(e))

    if settings.warmup_whisper:
        try:
            whisper_service.initialize()
        except Exception as e:
            logger.warning("Whisper warmup failed: %s", str(e))


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down backend")
    await redis_client.disconnect()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
