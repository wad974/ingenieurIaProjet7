"""Découpage, embeddings et index vectoriel FAISS."""

from __future__ import annotations

import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

logger = logging.getLogger(__name__)


# --- Découpage --------------------------------------------------------------
def decouper_texte(
    texte: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Découpe un texte en segments chevauchants prêts pour l'embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or config.CHUNK_OVERLAP,
    )
    return splitter.split_text(texte)


# --- Embeddings -------------------------------------------------------------
def embed_texts(
    textes: Sequence[str],
    batch_size: int | None = None,
    model: str | None = None,
) -> list[list[float]]:
    """Calcule les embeddings Ollama par lots."""
    textes = list(textes)
    if not textes:
        return []

    batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
    model = model or config.EMBEDDING_MODEL
    vecteurs: list[list[float]] = []

    logger.info("Calcul de %d embeddings avec %s", len(textes), model)
    for debut in range(0, len(textes), batch_size):
        lot = textes[debut : debut + batch_size]
        reponse = ollama.embed(model=model, input=lot)
        vecteurs.extend(reponse["embeddings"])
        logger.debug(
            "Embeddings %d-%d / %d",
            debut,
            debut + len(lot),
            len(textes),
        )

    if len(vecteurs) != len(textes):
        raise RuntimeError(
            "Nombre d'embeddings incohérent : "
            f"{len(vecteurs)} vecteurs pour {len(textes)} textes"
        )
    return vecteurs


def vectoriser_requete(question: str) -> np.ndarray:
    """Embedding normalisé L2 d'une question (cosinus via produit scalaire)."""
    vecteur = np.asarray(embed_texts([question]), dtype="float32")
    faiss.normalize_L2(vecteur)
    return vecteur


# --- Index FAISS ------------------------------------------------------------
def construire_index(
    chunks: Sequence[str],
    sources: Sequence[int],
    metadonnees: Sequence[dict[str, Any]],
    index_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> faiss.Index:
    """Calcule les vecteurs, construit l'index FAISS et le sauvegarde sur disque."""
    if len(chunks) != len(sources):
        raise ValueError("chunks et sources doivent avoir la même longueur")
    if not chunks:
        raise ValueError("Impossible de construire un index sans chunk")

    index_path = Path(index_path) if index_path else config.FAISS_INDEX_PATH
    metadata_path = Path(metadata_path) if metadata_path else config.METADATA_PATH

    vecteurs = np.asarray(embed_texts(chunks), dtype="float32")
    faiss.normalize_L2(vecteurs)  # cosinus via produit scalaire

    index = faiss.IndexFlatIP(vecteurs.shape[1])
    index.add(vecteurs)

    if not (index.ntotal == len(chunks) == len(sources)):
        raise RuntimeError("Index FAISS désaligné avec les chunks/sources")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with metadata_path.open("wb") as fichier:
        pickle.dump(
            {
                "chunks": list(chunks),
                "sources": list(sources),
                "metadata": list(metadonnees),
            },
            fichier,
        )

    charger_index.cache_clear()
    logger.info(
        "%d chunks indexés (%d dimensions) -> %s",
        index.ntotal,
        vecteurs.shape[1],
        index_path,
    )
    return index


@lru_cache(maxsize=1)
def charger_index(
    index_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[faiss.Index, dict[str, Any]]:
    """Charge l'index FAISS et ses métadonnées (mis en cache)."""
    index_path = Path(index_path) if index_path else config.FAISS_INDEX_PATH
    metadata_path = Path(metadata_path) if metadata_path else config.METADATA_PATH

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Index introuvable ({index_path}). "
            "Lance d'abord : python indexer.py"
        )

    index = faiss.read_index(str(index_path))
    with metadata_path.open("rb") as fichier:
        data = pickle.load(fichier)

    if not (index.ntotal == len(data["chunks"]) == len(data["sources"])):
        raise ValueError(
            "Index FAISS et chunks désalignés : "
            "relance l'ingestion avec python indexer.py"
        )
    return index, data


# --- Recherche --------------------------------------------------------------
def rechercher_evenements(
    question: str,
    k: int | None = None,
    score_min: float | None = None,
    index_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Retourne les chunks les plus proches de la question."""
    k = k or config.TOP_K
    score_min = config.SCORE_MIN if score_min is None else score_min

    index, data = charger_index(index_path, metadata_path)
    if index.ntotal == 0:
        return []

    scores, indices = index.search(
        vectoriser_requete(question),
        min(k, index.ntotal),
    )

    resultats: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or score < score_min:  # filtre de pertinence
            continue
        resultats.append(
            {
                "chunk": data["chunks"][idx],
                "evenement": data["metadata"][data["sources"][idx]],
                "score": float(score),
            }
        )

    logger.info("%d résultat(s) retenu(s) pour %r", len(resultats), question)
    return resultats
