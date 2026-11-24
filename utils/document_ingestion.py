"""Orchestration de l'ingestion : Markdown -> chunks -> index FAISS."""

from __future__ import annotations

import logging
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config, document_store
from .vector_store import decouper_texte

logger = logging.getLogger(__name__)


@dataclass
class SourceIngeree:
    """Une source déjà convertie en Markdown, prête à être indexée."""

    title: str
    source: str
    kind: str  # "file" ou "url"
    markdown: str


def _slug(texte: str) -> str:
    texte = unicodedata.normalize("NFKD", texte)
    texte = texte.encode("ascii", "ignore").decode("ascii")
    texte = re.sub(r"[^a-zA-Z0-9]+", "-", texte).strip("-").lower()
    return texte[:60] or "document"


def _vider_documents() -> None:
    config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    for chemin in config.DOCUMENTS_DIR.glob("*.md"):
        chemin.unlink(missing_ok=True)


def _sources_existantes() -> set[str]:
    if not document_store.index_existe():
        return set()
    try:
        _, metadata = document_store.charger_index_documents()
    except Exception:  # index corrompu : on laisse l'ingestion le signaler
        logger.exception("Impossible de lire les métadonnées existantes")
        return set()
    return {
        str(document.get("source"))
        for document in metadata.get("documents", [])
        if document.get("source")
    }


def _ecrire_markdown(
    source: SourceIngeree,
    identifiant: str,
) -> Path:
    config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    chemin = config.DOCUMENTS_DIR / f"{_slug(source.title)}-{identifiant}.md"
    chemin.write_text(source.markdown, encoding="utf-8")
    return chemin


def ingerer_sources(
    sources: list[SourceIngeree],
    replace: bool = False,
) -> dict[str, Any]:
    """Convertit les sources en chunks, les vectorise et met à jour l'index.

    Retourne un bilan par source et les statistiques de l'index.
    """
    if replace:
        _vider_documents()
        document_store.reinitialiser_index()

    deja_indexees = set() if replace else _sources_existantes()

    chunks: list[str] = []
    sources_par_chunk: list[int] = []
    documents: list[dict[str, Any]] = []
    resultats: list[dict[str, Any]] = []

    for source in sources:
        base = {
            "source": source.source,
            "title": source.title,
            "kind": source.kind,
            "characters": len(source.markdown),
            "markdown_path": None,
        }

        if source.source in deja_indexees:
            resultats.append(
                {
                    **base,
                    "status": "skipped",
                    "chunks": 0,
                    "error": "Source déjà indexée (utilise replace=true).",
                }
            )
            continue

        morceaux = decouper_texte(source.markdown)
        if not morceaux:
            resultats.append(
                {
                    **base,
                    "status": "error",
                    "chunks": 0,
                    "error": "Aucun chunk produit à partir du Markdown.",
                }
            )
            continue

        identifiant = uuid.uuid4().hex
        chemin = _ecrire_markdown(source, identifiant)
        try:
            chemin_relatif = str(chemin.relative_to(config.BASE_DIR))
        except ValueError:
            chemin_relatif = str(chemin)

        numero_document = len(documents)
        documents.append(
            {
                "id": identifiant,
                "title": source.title,
                "source": source.source,
                "kind": source.kind,
                "markdown_path": chemin_relatif,
                "characters": len(source.markdown),
                "chunks": len(morceaux),
            }
        )
        for morceau in morceaux:
            chunks.append(morceau)
            sources_par_chunk.append(numero_document)

        resultats.append(
            {
                **base,
                "status": "ok",
                "chunks": len(morceaux),
                "markdown_path": chemin_relatif,
                "error": None,
            }
        )

    if chunks:
        index_stats = document_store.ajouter_au_index(
            chunks,
            sources_par_chunk,
            documents,
            replace=replace,
        )
    else:
        index_stats = document_store.statistiques_index()

    logger.info(
        "Ingestion terminée : %d source(s) traitée(s), %d chunk(s) ajouté(s)",
        len(resultats),
        len(chunks),
    )
    return {
        "replace": replace,
        "sources": resultats,
        "index": index_stats,
    }
