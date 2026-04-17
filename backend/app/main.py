from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api import auth, sessions, chat
from app.services.mongo import ensure_indexes
from app.services.qdrant_client import ensure_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_indexes()
    ensure_collection()
    yield


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title="NyayaBot API", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[s.frontend_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router)
    app.include_router(sessions.router)
    app.include_router(chat.router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
