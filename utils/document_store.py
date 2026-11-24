"""Index vectoriel FAISS dédié aux documents ingérés via l'API.

Contrairement à l'index des événements (``bdd/faiss_index.idx``), cet index
est générique : il stocke les chunks Markdown et les documents dont ils
proviennent. Les vecteurs sont calculés avec Ollama et stockés uniquement
dans FAISS.
"""

from __future__ import annotations

import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np

from . import config, vector_store

logger = logging.getLogger(__name__)

# Ces globals sont volontairement remplaçables dans les tests.
INDEX_PATH: Path = config.RAG_INDEX_PATH
METADATA_PATH: Path = config.RAG_METADATA_PATH


def index_existe() -> bool:
    """Indique si un index RAG et ses métadonnées sont présents."""
    return INDEX_PATH.exists() and METADATA_PATH.exists()


def _sauvegarder(index: faiss.Index, metadata: dict[str, Any]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with METADATA_PATH.open("wb") as fichier:
        pickle.dump(metadata, fichier)


def reinitialiser_index() -> None:
    """Supprime l'index RAG et son cache mémoire."""
    for chemin in (INDEX_PATH, METADATA_PATH):
        chemin.unlink(missing_ok=True)
    charger_index_documents.cache_clear()
    logger.info("Index RAG réinitialisé")


@lru_cache(maxsize=1)
def charger_index_documents(
    index_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[faiss.Index, dict[str, Any]]:
    """Charge l'index RAG et ses métadonnées (mis en cache)."""
    chemin_index = Path(index_path) if index_path else INDEX_PATH
    chemin_meta = Path(metadata_path) if metadata_path else METADATA_PATH

    if not chemin_index.exists() or not chemin_meta.exists():
        raise FileNotFoundError(
            "Index RAG introuvable. Injecte d'abord des documents via "
            "POST /ingest."
        )

    index = faiss.read_index(str(chemin_index))
    with chemin_meta.open("rb") as fichier:
        metadata = pickle.load(fichier)

    if not (
        index.ntotal
        == len(metadata["chunks"])
        == len(metadata["sources"])
    ):
        raise ValueError(
            "Index RAG et métadonnées désalignés : relance une ingestion "
            "avec replace=true."
        )
    return index, metadata


def ajouter_au_index(
    chunks: Sequence[str],
    sources: Sequence[int],
    documents: Sequence[dict[str, Any]],
    replace: bool = False,
) -> dict[str, Any]:
    """Vectorise les chunks et les ajoute à l'index FAISS (ou le reconstruit)."""
    if len(chunks) != len(sources):
        raise ValueError("chunks et sources doivent avoir la même longueur")
    if not chunks:
        raise ValueError("Aucun chunk à indexer")

    vecteurs = np.asarray(
        vector_store.embed_texts(chunks),
        dtype="float32",
    )
    faiss.normalize_L2(vecteurs)

    if replace or not index_existe():
        index = faiss.IndexFlatIP(vecteurs.shape[1])
        metadata: dict[str, Any] = {
            "chunks": [],
            "sources": [],
            "documents": [],
        }
        decalage = 0
    else:
        index, metadata = charger_index_documents()
        if index.d != vecteurs.shape[1]:
            raise ValueError(
                "Dimension d'embedding incompatible avec l'index existant "
                f"({vecteurs.shape[1]} != {index.d}). Utilise replace=true."
            )
        decalage = len(metadata["documents"])

    index.add(vecteurs)
    metadata["chunks"].extend(list(chunks))
    metadata["sources"].extend(int(source) + decalage for source in sources)
    metadata["documents"].extend(list(documents))

    _sauvegarder(index, metadata)
    charger_index_documents.cache_clear()

    logger.info(
        "Index RAG : +%d chunks (%d documents au total, %d vecteurs)",
        len(chunks),
        len(metadata["documents"]),
        index.ntotal,
    )
    return {
        "exists": True,
        "documents": len(metadata["documents"]),
        "chunks": len(metadata["chunks"]),
        "dimension": index.d,
        "added_chunks": len(chunks),
    }


def rechercher_documents(
    question: str,
    k: int | None = None,
    score_min: float | None = None,
) -> list[dict[str, Any]]:
    """Recherche les chunks Markdown les plus proches d'une question."""
    if not index_existe():
        logger.warning("Recherche demandée sans index RAG : retour vide")
        return []

    k = k or config.TOP_K
    score_min = config.SCORE_MIN if score_min is None else score_min
    index, metadata = charger_index_documents()

    if index.ntotal == 0:
        return []

    scores, indices = index.search(
        vector_store.vectoriser_requete(question),
        min(k, index.ntotal),
    )

    resultats: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or score < score_min:
            continue
        source = metadata["sources"][idx]
        resultats.append(
            {
                "chunk": metadata["chunks"][idx],
                "document": metadata["documents"][source],
                "score": float(score),
            }
        )

    logger.info("%d chunk(s) retenu(s) pour %r", len(resultats), question)
    return resultats


def statistiques_index() -> dict[str, Any]:
    """Retourne l'état de l'index RAG (nombre de documents, chunks, dimension)."""
    if not index_existe():
        return {
            "exists": False,
            "documents": 0,
            "chunks": 0,
            "dimension": 0,
            "updated_at": None,
        }

    with METADATA_PATH.open("rb") as fichier:
        metadata = pickle.load(fichier)
    index = faiss.read_index(str(INDEX_PATH))

    updated_at = None
    if METADATA_PATH.exists():
        from datetime import datetime

        updated_at = datetime.fromtimestamp(
            METADATA_PATH.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "exists": True,
        "documents": len(metadata.get("documents", [])),
        "chunks": len(metadata.get("chunks", [])),
        "dimension": index.d,
        "updated_at": updated_at,
    }
