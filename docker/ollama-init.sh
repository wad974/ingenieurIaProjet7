#!/bin/sh
# Initialise les modèles Ollama dans le volume partagé (un seul pull).
# Attendu au premier démarrage ; ensuite le volume ollama_models suffit.
set -e

echo "Pull du modèle LLM : ${OLLAMA_MODEL}"
ollama pull "${OLLAMA_MODEL}"
echo "Pull du modèle d'embeddings : ${EMBEDDING_MODEL}"
ollama pull "${EMBEDDING_MODEL}"
echo "--- Modèles disponibles ---"
ollama list
