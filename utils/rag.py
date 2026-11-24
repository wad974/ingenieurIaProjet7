"""Génération de réponses RAG avec Ollama (LLM local uniquement)."""

from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from . import config, document_store

logger = logging.getLogger(__name__)

PROMPT = ChatPromptTemplate.from_template(
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

NON_TROUVE = (
    "Je ne trouve pas cette information dans les documents indexés."
)

_llm: ChatOllama | None = None
_chaine: Any = None
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


def obtenir_chaine():
    """Construit (une seule fois) la chaîne prompt -> LLM -> texte."""
    global _chaine
    if _chaine is None:
        llm = obtenir_llm()  # hors du verrou : éviter un deadlock non réentrant
        with _lock:
            if _chaine is None:
                _chaine = PROMPT | llm | StrOutputParser()
    return _chaine


def generer_reponse(question: str, contextes: list[str]) -> str:
    """Génère une réponse à partir des chunks fournis (sans appel si vide)."""
    contextes = [contexte for contexte in contextes if contexte.strip()]
    if not contextes:
        return NON_TROUVE

    context = "\n\n---\n\n".join(contextes)
    logger.info("Appel LLM %s pour %r", config.OLLAMA_MODEL, question)
    reponse = obtenir_chaine().invoke(
        {"context": context, "question": question}
    )
    return reponse.strip()


def repondre(
    question: str,
    top_k: int | None = None,
    score_min: float | None = None,
) -> dict[str, Any]:
    """Retrieval + génération en un seul appel (utilisé par les tests et l'API)."""
    contextes = document_store.rechercher_documents(
        question,
        k=top_k,
        score_min=score_min,
    )
    reponse = generer_reponse(
        question,
        [contexte["chunk"] for contexte in contextes],
    )
    return {"answer": reponse, "contexts": contextes}
