"""Génération de réponses RAG ou directes avec Ollama.

Deux modes, comme dans le cours OpenClassrooms :
- ``RAG`` : la réponse s'appuie uniquement sur les chunks FAISS retrouvés ;
- ``DIRECT`` : le LLM répond avec ses connaissances générales, sans retrieval.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from . import config, document_store
from .query_classifier import Classification, classifier_demande

logger = logging.getLogger(__name__)

PROMPT_RAG = ChatPromptTemplate.from_template(
    """
Tu es un assistant documentaire. Tu réponds uniquement à partir des extraits
fournis ci-dessous.

Extraits :
{context}

Question : {question}

Consignes :
- Réponds en français, de façon concise et structurée.
- Si les extraits ne contiennent pas la réponse, dis-le explicitement.
- N'invente aucune information.
- Cite les titres des documents utilisés quand c'est possible.
"""
)

PROMPT_DIRECT = ChatPromptTemplate.from_template(
    """
Tu es un assistant culturel de La Réunion.
Réponds à la question en utilisant tes connaissances générales, en français,
de façon concise et utile.
Si tu ne connais pas une information spécifique, dis-le clairement.
N'invente aucune information.

Question : {question}
"""
)

NON_TROUVE = (
    "Je n'ai trouvé aucun document pertinent dans la base de connaissances "
    "pour répondre à cette question."
)

_llm: ChatOllama | None = None
_chaine_rag: Any = None
_chaine_directe: Any = None
_lock = threading.Lock()


def obtenir_llm() -> ChatOllama:
    """Construit (une seule fois) le client LLM Ollama."""
    global _llm
    if _llm is None:
        with _lock:
            if _llm is None:
                _llm = ChatOllama(
                    model=config.OLLAMA_MODEL,
                    temperature=config.OLLAMA_TEMPERATURE,
                    num_ctx=config.OLLAMA_NUM_CTX,
                    num_predict=config.OLLAMA_NUM_PREDICT,
                    base_url=config.OLLAMA_BASE_URL,
                    reasoning=config.OLLAMA_REASONING,
                )
    return _llm


def obtenir_chaine_rag():
    """Construit (une seule fois) la chaîne RAG."""
    global _chaine_rag
    if _chaine_rag is None:
        llm = obtenir_llm()  # hors du verrou : éviter un deadlock non réentrant
        with _lock:
            if _chaine_rag is None:
                _chaine_rag = PROMPT_RAG | llm | StrOutputParser()
    return _chaine_rag


def obtenir_chaine_directe():
    """Construit (une seule fois) la chaîne de réponse directe."""
    global _chaine_directe
    if _chaine_directe is None:
        llm = obtenir_llm()
        with _lock:
            if _chaine_directe is None:
                _chaine_directe = PROMPT_DIRECT | llm | StrOutputParser()
    return _chaine_directe


def generer_reponse(
    question: str,
    contextes: list[str],
    mode: str = "RAG",
) -> str:
    """Génère une réponse en mode RAG ou DIRECT."""
    mode = mode.upper()

    if mode == "DIRECT":
        logger.info("Génération DIRECTE (sans RAG) pour %r", question)
        return obtenir_chaine_directe().invoke({"question": question}).strip()

    contextes = [
        contexte for contexte in contextes if contexte.strip()
    ]
    if not contextes:
        return NON_TROUVE

    context = "\n\n---\n\n".join(contextes)
    logger.info("Génération RAG avec %d contexte(s) pour %r", len(contextes), question)
    return obtenir_chaine_rag().invoke(
        {"context": context, "question": question}
    ).strip()


def repondre(
    question: str,
    top_k: int | None = None,
    score_min: float | None = None,
    mode: str = "auto",
) -> dict[str, Any]:
    """Classifie la question, récupère si besoin, puis génère la réponse."""
    mode = mode.lower()

    if mode == "rag":
        classification = Classification(True, 1.0, "Mode RAG forcé")
    elif mode == "direct":
        classification = Classification(False, 1.0, "Mode DIRECT forcé")
    elif mode == "auto":
        classification = classifier_demande(question)
    else:
        raise ValueError("mode doit valoir 'auto', 'rag' ou 'direct'")

    contextes: list[dict[str, Any]] = []
    if classification.needs_rag:
        contextes = document_store.rechercher_documents(
            question,
            k=top_k,
            score_min=score_min,
        )

    reponse = generer_reponse(
        question,
        [contexte["chunk"] for contexte in contextes],
        mode=classification.mode,
    )

    return {
        "answer": reponse,
        "contexts": contextes,
        "mode": classification.mode,
        "rag_used": classification.needs_rag,
        "confidence": classification.confidence,
        "reason": classification.reason,
    }
