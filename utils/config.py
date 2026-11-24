"""Configuration centrale du projet.

Toutes les valeurs modifiables sont lues depuis le fichier ``.env`` (ou depuis
l'environnement du système) avec des valeurs par défaut raisonnables.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Emplacements -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)


def _str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "oui", "on"}


# --- Chemins ----------------------------------------------------------------
DATA_DIR = BASE_DIR / "bdd"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.idx"
METADATA_PATH = DATA_DIR / "metadata.pkl"
FEEDBACK_DB_PATH = DATA_DIR / "feedback.db"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = _str("LOG_FILE", str(LOG_DIR / "app.log"))

# Index RAG alimenté par l'API (fichiers et URLs convertis en Markdown)
DOCUMENTS_DIR = BASE_DIR / "documents"
RAG_INDEX_PATH = DATA_DIR / "rag_index.faiss"
RAG_METADATA_PATH = DATA_DIR / "rag_metadata.pkl"
MAX_UPLOAD_BYTES = _int("MAX_UPLOAD_BYTES", 20 * 1024 * 1024)

# --- Source de données (OpenAgenda / OpenDataSoft) --------------------------
OPENAGENDA_URL = _str(
    "OPENAGENDA_URL",
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "evenements-publics-openagenda/records",
)
DATE_MIN = _str("DATE_MIN", "2026-01-01T00:00:00")
MAX_EVENEMENTS = _int("MAX_EVENEMENTS", 500)
PAGE_SIZE = _int("PAGE_SIZE", 100)
REQUEST_TIMEOUT = _int("REQUEST_TIMEOUT", 60)

# --- Découpage et embeddings ------------------------------------------------
CHUNK_SIZE = _int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 100)
EMBEDDING_MODEL = _str("EMBEDDING_MODEL", "embeddinggemma-2:latest")
EMBEDDING_BATCH_SIZE = _int("EMBEDDING_BATCH_SIZE", 32)

# --- Recherche vectorielle --------------------------------------------------
TOP_K = _int("TOP_K", 4)
# Les similarités cosinus d'embeddinggemma-2 se situent autour de 0.70-0.76
# sur ce corpus : un seuil plus haut filtrerait tous les résultats.
SCORE_MIN = _float("SCORE_MIN", 0.70)

# --- LLM local (Ollama uniquement) ------------------------------------------
OLLAMA_MODEL = _str("OLLAMA_MODEL", "qwen3.5:2b")
OLLAMA_BASE_URL = _str("OLLAMA_BASE_URL", "") or None
OLLAMA_TEMPERATURE = _float("OLLAMA_TEMPERATURE", 0.2)
OLLAMA_NUM_CTX = _int("OLLAMA_NUM_CTX", 8192)
OLLAMA_NUM_PREDICT = _int("OLLAMA_NUM_PREDICT", 512)
# qwen3.5 est un modèle "thinking" : laisser la réflexion active consomme
# tous les tokens et renvoie une réponse vide. On la désactive par défaut.
OLLAMA_REASONING = _bool("OLLAMA_REASONING", False)

# --- Journalisation ---------------------------------------------------------
LOG_LEVEL = _str("LOG_LEVEL", "INFO").upper()


def ensure_directories() -> None:
    """Crée les dossiers de données, documents et logs s'ils n'existent pas."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
