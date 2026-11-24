"""Flux 3 — état de l'index et évaluation des métriques."""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import IndexStats, MetricsOut, MetricsRequest
from api.services.evaluation import evaluer
from utils import document_store

router = APIRouter(tags=["3. Métriques"])


@router.get(
    "/metrics",
    response_model=IndexStats,
    summary="État de l'index RAG",
)
def etat_index() -> IndexStats:
    return IndexStats(**document_store.statistiques_index())


@router.post(
    "/metrics",
    response_model=MetricsOut,
    summary="Évaluer le retrieval et la génération",
    description=(
        "Calcule hit@k, precision@k, recall@k et MRR à partir de "
        "`expected_sources` ou `expected_snippets`. Si `include_generation` "
        "est vrai, calcule aussi le F1 (si `expected_answer` est fourni) et "
        "la groundedness de la réponse."
    ),
)
def metriques(payload: MetricsRequest) -> MetricsOut:
    return evaluer(payload)
