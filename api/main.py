"""Application FastAPI — API RAG à 3 flux (Ollama + FAISS).

Lancement :
    uvicorn api.main:app --reload --port 8000

Documentation interactive : http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import ingestion, metrics, prompt
from utils import config, document_store
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    config.ensure_directories()
    logger.info(
        "API RAG démarrée — LLM %s, embeddings %s",
        config.OLLAMA_MODEL,
        config.EMBEDDING_MODEL,
    )
    yield
    logger.info("API RAG arrêtée")


app = FastAPI(
    title="API RAG — Ollama + FAISS",
    description=(
        "Trois flux : prompt et réponse fusionnés (une question -> une réponse "
        "JSON), ingestion de fichiers/URLs (Markdown -> chunks -> embeddings "
        "-> FAISS) et métriques."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prompt.router)
app.include_router(ingestion.router)
app.include_router(metrics.router)


@app.get("/", tags=["Santé"])
def racine() -> dict:
    return {
        "name": "API RAG — Ollama + FAISS",
        "docs": "/docs",
        "endpoints": {
            "flux_1_prompt_reponse": "POST /prompt",
            "flux_2_ingestion": "POST /ingest",
            "flux_3_metriques": "POST /metrics (GET /metrics pour l'état)",
        },
    }


@app.get("/health", tags=["Santé"])
def health() -> dict:
    return {
        "status": "ok",
        "provider": "ollama",
        "llm_model": config.OLLAMA_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "index": document_store.statistiques_index(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
