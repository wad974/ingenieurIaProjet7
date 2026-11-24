"""Évaluation du retrieval et de la génération (flux 3)."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from api.schemas import EvalCase, MetricsOut, MetricsRequest
from utils import document_store
from utils.rag import generer_reponse

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokens(texte: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(texte or "")]


def _normaliser(texte: str) -> str:
    return " ".join(_tokens(texte))


def _f1(prediction: str, reference: str) -> float:
    """F1 sur les tokens (bag-of-words), entre 0 et 1."""
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred or not ref:
        return 0.0

    commun = Counter(pred) & Counter(ref)
    overlap = sum(commun.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _document_correspond(
    document: dict[str, Any],
    attendus: list[str],
) -> bool:
    contenu = " ".join(
        str(document.get(cle) or "")
        for cle in ("id", "title", "source", "markdown_path")
    ).lower()
    return any(attendu.lower() in contenu for attendu in attendus if attendu)


def _evaluer_retrieval(
    retrieved: list[dict[str, Any]],
    case: EvalCase,
) -> dict[str, float] | None:
    """Calcule hit@k, precision@k, recall@k et MRR si un attendu est fourni."""
    attendus = [attendu for attendu in case.expected_sources if attendu]
    snippets = [snippet for snippet in case.expected_snippets if snippet]

    if attendus:
        documents = [resultat["document"] for resultat in retrieved]
        pertinents = [
            any(_document_correspond(document, [attendu]) for attendu in attendus)
            for document in documents
        ]
        trouves = {
            attendu
            for attendu in attendus
            if any(_document_correspond(document, [attendu]) for document in documents)
        }
        nb_attendus = len(attendus)
    elif snippets:
        chunks = [resultat["chunk"] for resultat in retrieved]
        pertinents = [
            any(snippet.lower() in chunk.lower() for snippet in snippets)
            for chunk in chunks
        ]
        trouves = {
            snippet
            for snippet in snippets
            if any(snippet.lower() in chunk.lower() for chunk in chunks)
        }
        nb_attendus = len(snippets)
    else:
        return None

    if not retrieved:
        return {"hit@k": 0.0, "precision@k": 0.0, "recall@k": 0.0, "mrr": 0.0}

    nb_pertinents = sum(1 for pertinent in pertinents if pertinent)
    hit = 1.0 if nb_pertinents else 0.0
    precision = nb_pertinents / len(pertinents) if pertinents else 0.0
    recall = min(1.0, len(trouves) / nb_attendus) if nb_attendus else 0.0

    mrr = 0.0
    for rang, pertinent in enumerate(pertinents, start=1):
        if pertinent:
            mrr = 1.0 / rang
            break

    return {
        "hit@k": hit,
        "precision@k": precision,
        "recall@k": recall,
        "mrr": mrr,
    }


def _groundedness(reponse: str, contextes: list[str]) -> float:
    """Part des mots significatifs de la réponse présents dans les contextes."""
    mots = [token for token in _tokens(reponse) if len(token) > 3]
    if not mots:
        return 0.0
    vocabulaire = set(_tokens(" ".join(contextes)))
    return sum(1 for mot in mots if mot in vocabulaire) / len(mots)


def _moyenne(valeurs: list[float]) -> float | None:
    if not valeurs:
        return None
    return round(sum(valeurs) / len(valeurs), 4)


def evaluer(payload: MetricsRequest) -> MetricsOut:
    """Évalue chaque cas puis agrège les métriques."""
    index_stats = document_store.statistiques_index()
    resultats: list[dict[str, Any]] = []
    metriques_retrieval: list[dict[str, float]] = []
    scores_f1: list[float] = []
    scores_em: list[float] = []
    scores_groundedness: list[float] = []

    for case in payload.cases:
        retrieved = document_store.rechercher_documents(
            case.question,
            k=payload.top_k,
            score_min=payload.score_min,
        )
        chunks = [resultat["chunk"] for resultat in retrieved]

        metriques_cas: dict[str, Any] = {}

        retrieval = _evaluer_retrieval(retrieved, case)
        if retrieval is not None:
            metriques_retrieval.append(retrieval)
            metriques_cas.update(
                {cle: round(valeur, 4) for cle, valeur in retrieval.items()}
            )

        reponse: str | None = None
        if payload.include_generation:
            reponse = generer_reponse(case.question, chunks)
            metriques_cas["groundedness"] = round(
                _groundedness(reponse, chunks),
                4,
            )
            scores_groundedness.append(metriques_cas["groundedness"])

            if case.expected_answer:
                f1 = _f1(reponse, case.expected_answer)
                em = (
                    1.0
                    if _normaliser(reponse) == _normaliser(case.expected_answer)
                    else 0.0
                )
                scores_f1.append(f1)
                scores_em.append(em)
                metriques_cas["f1"] = round(f1, 4)
                metriques_cas["exact_match"] = em
        elif case.expected_answer:
            metriques_cas["f1"] = None
            metriques_cas["exact_match"] = None
            metriques_cas["note"] = (
                "include_generation=true requis pour les métriques de réponse."
            )

        resultats.append(
            {
                "question": case.question,
                "retrieved": [
                    {
                        "score": round(resultat["score"], 4),
                        "title": resultat["document"].get("title"),
                        "source": resultat["document"].get("source"),
                    }
                    for resultat in retrieved
                ],
                "metrics": metriques_cas,
                "answer": reponse,
            }
        )

    aggregate: dict[str, Any] = {
        "hit@k": _moyenne([m["hit@k"] for m in metriques_retrieval]),
        "precision@k": _moyenne([m["precision@k"] for m in metriques_retrieval]),
        "recall@k": _moyenne([m["recall@k"] for m in metriques_retrieval]),
        "mrr": _moyenne([m["mrr"] for m in metriques_retrieval]),
        "f1": _moyenne(scores_f1),
        "exact_match": _moyenne(scores_em),
        "groundedness": _moyenne(scores_groundedness),
        "n_cases": len(payload.cases),
        "n_retrieval_evalues": len(metriques_retrieval),
        "n_generation_evalues": len(scores_groundedness),
    }

    logger.info(
        "Évaluation terminée : %d cas, %d avec métriques de retrieval",
        len(payload.cases),
        len(metriques_retrieval),
    )
    return MetricsOut(
        n_cases=len(payload.cases),
        index=index_stats,
        aggregate=aggregate,
        results=resultats,
    )
