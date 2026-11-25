#!/bin/sh
# Point d'entrée du conteneur API : dossiers puis uvicorn.
set -e

python -c "from utils import config; config.ensure_directories()"

echo "API RAG — Ollama ${OLLAMA_BASE_URL:-http://localhost:11434}, \
LLM ${OLLAMA_MODEL:?}, embeddings ${EMBEDDING_MODEL:?}"
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
