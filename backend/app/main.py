import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api import auth, sessions, chat, documents
from app.services.mongo import ensure_indexes
from app.services.qdrant_client import ensure_collection, ensure_user_docs_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

log = logging.getLogger("nyayabot.startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_indexes()
    try:
        ensure_collection()
        ensure_user_docs_collection()
    except Exception as exc:
        log.warning("Qdrant unreachable at startup (cluster may be hibernated): %s", exc)
        log.warning("Chat will fail until Qdrant is reachable. Resume the cluster at cloud.qdrant.io")
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
    app.include_router(documents.router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
