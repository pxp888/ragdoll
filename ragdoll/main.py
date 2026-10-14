import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

from ragdoll.config import settings
from ragdoll.database import init_db
from ragdoll.routes import router
from ragdoll.clients.embedding import EmbeddingClient
from ragdoll.clients.kev import KevClient
from ragdoll.clients.standard_reranker import StandardReranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ragdoll")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize SQLite schema & storage directories
    logger.info(f"Initializing database at {settings.db_path}")
    init_db(settings.db_path)
    settings.storage_path.mkdir(parents=True, exist_ok=True)

    # 2. Initialize external clients if not already injected
    if not hasattr(app.state, "embedding_client") or app.state.embedding_client is None:
        app.state.embedding_client = EmbeddingClient(
            base_url=settings.embed_url,
            model=settings.embed_model,
            timeout=settings.request_timeout,
        )
    if not hasattr(app.state, "kev_client") or app.state.kev_client is None:
        app.state.kev_client = KevClient(
            base_url=settings.kev_url,
            auth_token=settings.kev_auth_token,
            timeout=settings.request_timeout,
        )
    if not hasattr(app.state, "standard_reranker") or app.state.standard_reranker is None:
        app.state.standard_reranker = StandardReranker(
            model_name=settings.reranker_model,
            cache_dir=settings.reranker_cache_dir,
        )

    embedding_client = app.state.embedding_client
    kev_client = app.state.kev_client

    # 3. Startup probe (Strict fail-fast policy)
    # The server verifies connectivity to both external services on startup.
    # If either dependency is unreachable, startup aborts immediately.
    logger.info("Executing startup connectivity probes to external dependencies...")
    embed_ok = False
    kev_ok = False

    try:
        embed_ok = await embedding_client.check_health()
    except Exception as e:
        logger.error(f"Startup probe: EmbeddingGemma2 unreachable at {settings.embed_url}: {e}")

    try:
        kev_ok = await kev_client.check_health()
    except Exception as e:
        logger.error(f"Startup probe: Kev 4B unreachable at {settings.kev_url}: {e}")

    # Check if startup probe should strictly abort
    skip_probe = getattr(settings, "skip_startup_probe", False)
    if not (embed_ok and kev_ok) and not skip_probe:
        err_msg = (
            f"FATAL: Dependency startup probe failed! "
            f"EmbeddingGemma2={'OK' if embed_ok else 'FAILED'}, "
            f"Kev-4B={'OK' if kev_ok else 'FAILED'}. "
            f"Ragdoll enforces a strict fail-fast policy and will not start with degraded dependencies."
        )
        logger.critical(err_msg)
        raise RuntimeError(err_msg)

    logger.info("All dependency probes passed successfully. ragdoll is ready to serve.")
    yield
    logger.info("Shutting down ragdoll server.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ragdoll",
        description="RAG Server with Multi-Scale Indexing and Dual Kev / Standard Retrieval",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()


def cli():
    uvicorn.run(
        "ragdoll.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    cli()
