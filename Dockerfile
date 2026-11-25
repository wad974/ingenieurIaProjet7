# Image de l'API RAG — Ollama + FAISS
# Build :  docker compose build
# Run  :  docker compose up

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
# Dépendances d'abord : ce layer ne change que si requirements.txt change.
# `g++` n'est nécessaire que pour compiler scikit-network (dépendance de
# RAGAS, pas de wheel cp314) ; il est retiré juste après pour alléger l'image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y g++ \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Code applicatif
COPY api/ api/
COPY utils/ utils/
COPY indexer.py .

# Point d'entrée : prépare les dossiers puis lance uvicorn sur 0.0.0.0:8000
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

# Pas de HEALTHCHECK ici : il dépend du port publié, géré dans docker-compose.yml
ENTRYPOINT ["/entrypoint.sh"]
