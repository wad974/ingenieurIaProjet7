"""Évaluation RAGAS : faithfulness, answer_relevancy, context_precision,
context_recall.

Reprend l'approche du notebook du cours OpenClassrooms, adaptée à Ollama
(LLM + embeddings locaux) et à l'index FAISS du projet.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import pandas as pd
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from utils import config

logger = logging.getLogger(__name__)

METRIQUES_RAGAS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

METRIQUES_DISPONIBLES = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}

# RAGAS 0.4 déprécie ces wrappers au profit de llm_factory, mais ils restent
# la voie la plus simple avec Ollama (et correspondent au cours).
_llm: Any = None
_embeddings: Any = None
_lock = threading.Lock()


def _obtenir_llm() -> Any:
    global _llm
    if _llm is None:
        with _lock:
            if _llm is None:
                chat = ChatOllama(
                    model=config.OLLAMA_MODEL,
                    temperature=0,
                    num_ctx=config.OLLAMA_NUM_CTX,
                    num_predict=config.OLLAMA_NUM_PREDICT,
                    base_url=config.OLLAMA_BASE_URL,
                    reasoning=False,
                )
                _llm = LangchainLLMWrapper(chat)
    return _llm


def _obtenir_embeddings() -> Any:
    global _embeddings
    if _embeddings is None:
        with _lock:
            if _embeddings is None:
                embeddings = OllamaEmbeddings(
                    model=config.EMBEDDING_MODEL,
                    base_url=config.OLLAMA_BASE_URL,
                )
                _embeddings = LangchainEmbeddingsWrapper(embeddings)
    return _embeddings


def _valeur(valeur: Any) -> float | None:
    """Convertit une valeur RAGAS en float JSON-compatible (NaN -> None)."""
    if valeur is None:
        return None
    try:
        if pd.isna(valeur):
            return None
        return round(float(valeur), 4)
    except (TypeError, ValueError):
        return None


def evaluer_avec_ragas(
    echantillons: list[dict[str, Any]],
    metriques: list[str] | None = None,
) -> dict[str, Any]:
    """Évalue des échantillons avec les métriques RAGAS demandées.

    Chaque échantillon doit contenir : ``question``, ``answer``, ``contexts``
    (liste de chunks) et ``ground_truth`` (peut être vide si les métriques
    contextuelles ne sont pas demandées).
    """
    if not echantillons:
        return {"scores": {}, "details": []}

    noms = [
        nom
        for nom in (metriques or list(METRIQUES_RAGAS))
        if nom in METRIQUES_DISPONIBLES
    ]
    if not noms:
        return {"scores": {}, "details": []}

    dataset = Dataset.from_dict(
        {
            "question": [
                echantillon["question"] for echantillon in echantillons
            ],
            "answer": [
                echantillon["answer"] for echantillon in echantillons
            ],
            "contexts": [
                echantillon["contexts"] for echantillon in echantillons
            ],
            "ground_truth": [
                echantillon.get("ground_truth") or ""
                for echantillon in echantillons
            ],
        }
    )

    logger.info(
        "Évaluation RAGAS sur %d cas avec %s…",
        len(echantillons),
        ", ".join(noms),
    )
    debut = time.perf_counter()
    resultat = evaluate(
        dataset=dataset,
        metrics=[METRIQUES_DISPONIBLES[nom] for nom in noms],
        llm=_obtenir_llm(),
        embeddings=_obtenir_embeddings(),
        show_progress=False,
    )
    duree = time.perf_counter() - debut
    df = resultat.to_pandas()

    scores: dict[str, float | None] = {}
    for metrique in noms:
        if metrique not in df.columns:
            scores[metrique] = None
            continue
        valeurs = [
            float(valeur)
            for valeur in df[metrique].dropna().tolist()
        ]
        scores[metrique] = (
            round(sum(valeurs) / len(valeurs), 4) if valeurs else None
        )

    details = [
        {
            metrique: _valeur(ligne.get(metrique))
            for metrique in noms
        }
        for _, ligne in df.iterrows()
    ]

    logger.info("RAGAS terminé en %.1f s : %s", duree, scores)
    return {"scores": scores, "details": details}
