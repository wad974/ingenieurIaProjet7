"""Métriques d'une réponse : retrieval, génération et RAGAS.

Ce module est appelé directement par le flux ``/prompt`` : les métriques sont
renvoyées dans la même réponse JSON que la réponse du LLM. Il n'y a plus
d'endpoint ``/metrics`` séparé.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

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
    attendus: list[str],
    snippets: list[str],
) -> dict[str, float] | None:
    """Calcule hit@k, precision@k, recall@k et MRR si un attendu est fourni."""
    attendus = [attendu for attendu in attendus if attendu]
    snippets = [snippet for snippet in snippets if snippet]

    if attendus:
        documents = [resultat["document"] for resultat in retrieved]
        pertinents = [
            any(
                _document_correspond(document, [attendu])
                for attendu in attendus
            )
            for document in documents
        ]
        trouves = {
            attendu
            for attendu in attendus
            if any(
                _document_correspond(document, [attendu])
                for document in documents
            )
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


def evaluer_reponse(
    question: str,
    answer: str,
    mode: str,
    contexts: list[dict[str, Any]],
    ground_truth: str | None = None,
    expected_sources: list[str] | None = None,
    expected_snippets: list[str] | None = None,
    include_ragas: bool = True,
) -> dict[str, Any]:
    """Calcule les métriques d'une réponse et les renvoie avec la réponse."""
    chunks = [contexte["chunk"] for contexte in contexts]
    metriques: dict[str, Any] = {}

    # --- Métriques de retrieval --------------------------------------------
    retrieval = _evaluer_retrieval(
        contexts,
        expected_sources or [],
        expected_snippets or [],
    )
    if retrieval is not None:
        metriques.update(
            {cle: round(valeur, 4) for cle, valeur in retrieval.items()}
        )

    # --- Métriques de génération -------------------------------------------
    if chunks:
        metriques["groundedness"] = round(
            _groundedness(answer, chunks),
            4,
        )

    if ground_truth:
        f1 = _f1(answer, ground_truth)
        metriques["f1"] = round(f1, 4)
        metriques["exact_match"] = (
            1.0
            if _normaliser(answer) == _normaliser(ground_truth)
            else 0.0
        )

    # --- Métriques RAGAS ----------------------------------------------------
    ragas_error: str | None = None
    if include_ragas:
        if not answer.strip():
            ragas_error = "Réponse vide : RAGAS ignoré."
        else:
            noms: list[str] = []
            if mode == "RAG" and chunks:
                noms.append("faithfulness")
            noms.append("answer_relevancy")
            if ground_truth and chunks:
                noms.extend(["context_precision", "context_recall"])

            try:
                from api.services.ragas_evaluation import (
                    evaluer_avec_ragas,
                )

                resultat = evaluer_avec_ragas(
                    [
                        {
                            "question": question,
                            "answer": answer,
                            "contexts": chunks,
                            "ground_truth": ground_truth or "",
                        }
                    ],
                    metriques=noms,
                )
                metriques.update(resultat["scores"])
            except Exception as exc:
                logger.exception("Échec de l'évaluation RAGAS")
                ragas_error = f"RAGAS a échoué : {exc}"

    return {"metrics": metriques, "ragas_error": ragas_error}
