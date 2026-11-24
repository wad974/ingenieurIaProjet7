"""Flux 1 — prompt, réponse et métriques dans une seule réponse JSON.

Une seule requête : `POST /prompt` avec `{"question": "..."}`.
La question est classifiée (RAG ou DIRECT), la réponse est générée puis les
métriques (retrieval, génération, RAGAS) sont renvoyées dans le même JSON.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from api.schemas import ContextOut, PromptRequest, PromptResponse
from api.services.evaluation import evaluer_reponse
from utils import config
from utils.rag import repondre

router = APIRouter(tags=["1. Prompt + Réponse + Métriques"])


@router.post(
    "/prompt",
    response_model=PromptResponse,
    summary="Poser une question : réponse RAG/DIRECT + métriques",
    description=(
        "Classe la question (RAG ou DIRECT), récupère les chunks FAISS si "
        "nécessaire, génère la réponse avec Ollama, calcule les métriques "
        "RAGAS et renvoie le tout dans une seule réponse JSON."
    ),
)
def poser_question(payload: PromptRequest) -> PromptResponse:
    debut = time.perf_counter()

    resultat = repondre(
        payload.question,
        top_k=payload.top_k,
        score_min=payload.score_min,
        mode=payload.mode,
    )

    metriques = None
    ragas_error = None
    metrics_elapsed_ms = None

    if payload.include_metrics:
        debut_metriques = time.perf_counter()
        evaluation = evaluer_reponse(
            question=payload.question,
            answer=resultat["answer"],
            mode=resultat["mode"],
            contexts=resultat["contexts"],
            ground_truth=payload.ground_truth,
            expected_sources=payload.expected_sources,
            expected_snippets=payload.expected_snippets,
            include_ragas=payload.include_ragas,
        )
        metriques = evaluation["metrics"]
        ragas_error = evaluation["ragas_error"]
        metrics_elapsed_ms = round(
            (time.perf_counter() - debut_metriques) * 1000,
            2,
        )

    ecoule = (time.perf_counter() - debut) * 1000
    return PromptResponse(
        question=payload.question,
        answer=resultat["answer"],
        mode=resultat["mode"],
        rag_used=resultat["rag_used"],
        confidence=resultat["confidence"],
        reason=resultat["reason"],
        contexts=[
            ContextOut(**contexte) for contexte in resultat["contexts"]
        ],
        model=config.OLLAMA_MODEL,
        elapsed_ms=round(ecoule, 2),
        metrics=metriques,
        ragas_error=ragas_error,
        metrics_elapsed_ms=metrics_elapsed_ms,
    )
