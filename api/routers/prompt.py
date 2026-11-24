"""Flux 1 — prompt et réponse fusionnés.

Une seule requête : `POST /prompt` avec `{"question": "..."}` renvoie
immédiatement la réponse générée, les contextes FAISS et le modèle utilisé.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from api.schemas import ContextOut, PromptRequest, PromptResponse
from utils import config, document_store
from utils.rag import generer_reponse

router = APIRouter(tags=["1. Prompt + Réponse"])


@router.post(
    "/prompt",
    response_model=PromptResponse,
    summary="Poser une question et obtenir la réponse en JSON",
    description=(
        "Recherche les chunks pertinents dans FAISS, génère la réponse avec "
        "Ollama et renvoie le tout en JSON : `question`, `answer`, "
        "`contexts`, `model`, `elapsed_ms`."
    ),
)
def poser_question(payload: PromptRequest) -> PromptResponse:
    debut = time.perf_counter()

    contextes = document_store.rechercher_documents(
        payload.question,
        k=payload.top_k,
        score_min=payload.score_min,
    )
    answer = generer_reponse(
        payload.question,
        [contexte["chunk"] for contexte in contextes],
    )

    ecoule = (time.perf_counter() - debut) * 1000
    return PromptResponse(
        question=payload.question,
        answer=answer,
        contexts=[ContextOut(**contexte) for contexte in contextes],
        model=config.OLLAMA_MODEL,
        elapsed_ms=round(ecoule, 2),
    )
