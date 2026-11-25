#!/usr/bin/env bash
# Construit l'index vectoriel FAISS des événements (bdd/faiss_index.idx).
#
# Usage :
#   ./scripts/build_index.sh              # 500 événements (défaut .env)
#   ./scripts/build_index.sh 200          # 200 événements
#
# Prérequis : docker compose up -d (le service ollama doit tourner, les
# embeddings sont calculés par Ollama). La liste des événements est chargée
# depuis l'API publique OpenAgenda (connexion nécessaire uniquement ici).
set -euo pipefail
cd "$(dirname "$0")/.."

MAX=${1:-${MAX_EVENEMENTS:-500}}
echo "=== Build de l'index événements (${MAX} événements max) ==="
docker compose run --rm api python indexer.py --max-evenements "${MAX}"
echo "=== Terminé : bdd/faiss_index.idx + bdd/metadata.pkl ==="
