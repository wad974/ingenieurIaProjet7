# API RAG — Ollama + FAISS

API FastAPI à **3 flux** (prompt et réponse fusionnés) pour un assistant documentaire RAG :

| Flux | Endpoint | Rôle |
|------|----------|------|
| 1. Prompt + Réponse | `POST /prompt` | Reçoit la question (`BaseModel`), cherche dans FAISS, génère la réponse Ollama et renvoie **immédiatement** le JSON complet |
| 2. Ingestion | `POST /ingest` | Fichiers et/ou URLs → **Markdown** → chunks → embeddings → index FAISS sauvegardé |
| 3. Métriques | `POST /metrics` | Évalue hit@k, precision@k, recall@k, MRR, F1, exact match et groundedness |

- LLM et embeddings : **Ollama uniquement** (`qwen3.5:2b` + `embeddinggemma-2`).
- Recherche vectorielle : **FAISS uniquement** (`IndexFlatIP` + normalisation L2, similarité cosinus).
- Documentation interactive : `http://localhost:8000/docs`.

## Architecture

```
Projet7/
├── api/
│   ├── main.py                 # Application FastAPI, CORS, /health
│   ├── schemas.py              # Modèles Pydantic (BaseModel) d'entrée/sortie
│   ├── routers/
│   │   ├── prompt.py           # Flux 1 : POST /prompt (question -> réponse)
│   │   ├── ingestion.py        # Flux 2 : POST /ingest
│   │   └── metrics.py          # Flux 3 : POST /metrics + GET /metrics
│   └── services/
│       └── evaluation.py       # Calcul des métriques
├── utils/
│   ├── config.py               # Configuration (.env), chemins, modèles
│   ├── markdown_converter.py   # PDF/DOCX/TXT/MD/HTML/URL -> Markdown
│   ├── document_store.py       # Index FAISS des documents (recherche, ajout)
│   ├── document_ingestion.py   # Orchestration Markdown -> chunks -> index
│   ├── rag.py                  # Prompt + LLM Ollama (génération)
│   ├── vector_store.py         # Embeddings Ollama, découpage, index événements
│   ├── data_loader.py          # Chargement des événements OpenAgenda
│   ├── database.py             # Feedback utilisateurs (héritage, non utilisé)
│   └── logging_config.py       # Logging console + fichier rotatif
├── documents/                  # Markdown ingérés (généré, ignoré par git)
├── bdd/
│   ├── rag_index.faiss         # Index FAISS des documents (généré)
│   ├── rag_metadata.pkl        # Chunks + métadonnées documents (généré)
│   └── faiss_index.idx         # Index événements historique
├── indexer.py                  # Construction de l'index événements
├── test/                       # Tests unitaires + intégration API
├── requirements.txt
├── .env.example
└── .gitignore
```

## Installation

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# Modèles Ollama
ollama pull embeddinggemma-2:latest
ollama pull qwen3.5:2b

cp .env.example .env
uvicorn api.main:app --reload --port 8000
```

> `qwen3.5` est un modèle « thinking ». `OLLAMA_REASONING=false` est le
> réglage par défaut : avec la réflexion active, le modèle consomme tous ses
> tokens en interne et renvoie une réponse vide.

## Utilisation

### Flux 1 — poser une question et obtenir la réponse JSON

Le corps de la requête est un `BaseModel` `PromptRequest` avec un champ
`question` (visible comme input dans `/docs`) :

```bash
curl -X POST http://localhost:8000/prompt \
  -H "Content-Type: application/json" \
  -d '{"question": "Quel concert de jazz à Saint-Denis ?", "top_k": 4, "score_min": 0.70}'
```

Réponse JSON immédiate (`PromptResponse`) :

```json
{
  "question": "Quel concert de jazz à Saint-Denis ?",
  "answer": "Le concert de jazz aura lieu le 12 juillet 2026 à Saint-Denis, au parc de la Source.",
  "contexts": [
    {
      "chunk": "Le 12 juillet 2026, un concert de jazz…",
      "score": 0.804,
      "document": {"id": "…", "title": "programme", "source": "programme.pdf", "kind": "file"}
    }
  ],
  "model": "qwen3.5:2b",
  "elapsed_ms": 4612.3
}
```

La recherche FAISS et la génération Ollama sont exécutées dans la même
requête : un seul appel suffit pour obtenir la réponse.

### Flux 2 — injecter des fichiers et/ou des URLs

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@programme.pdf" \
  -F "files=@notes.docx" \
  -F "urls=https://exemple.com/article" \
  -F "replace=false"
```

- Formats fichiers : **PDF, DOCX, TXT, MD, HTML**.
- Une URL peut pointer vers une page HTML, un PDF ou un fichier texte.
- `/ingest` attend du **multipart/form-data** (pas du JSON).
- Un champ `files` vide (comportement de Swagger quand aucun fichier n'est
  sélectionné) est toléré : tu peux renseigner uniquement `urls`.
- `replace=true` vide l'index et le dossier `documents/` avant d'ingérer.
- Les sources déjà indexées sont ignorées (`status: "skipped"`).

Réponse (extrait) :

```json
{
  "replace": false,
  "sources": [
    {
      "source": "programme.pdf",
      "title": "programme",
      "kind": "file",
      "status": "ok",
      "characters": 4210,
      "chunks": 7,
      "markdown_path": "documents/programme-1a2b3c4d.md",
      "error": null
    }
  ],
  "index": {"exists": true, "documents": 1, "chunks": 7, "dimension": 768}
}
```

### Flux 3 — tester les métriques

```bash
# État de l'index
curl http://localhost:8000/metrics

# Évaluation
curl -X POST http://localhost:8000/metrics \
  -H "Content-Type: application/json" \
  -d '{
    "cases": [
      {
        "question": "Quel concert à Saint-Denis ?",
        "expected_sources": ["programme.pdf"],
        "expected_snippets": ["concert de jazz"]
      }
    ],
    "top_k": 4,
    "score_min": 0.70,
    "include_generation": true,
    "expected_answer": "Un concert de jazz le 12 juillet 2026."
  }'
```

Métriques calculées :

- **Retrieval** : `hit@k`, `precision@k`, `recall@k`, `mrr` — à partir de
  `expected_sources` (titre/id/source du document) ou `expected_snippets`
  (texte présent dans le chunk).
- **Génération** (`include_generation=true`) : `groundedness` (part des mots
  de la réponse présents dans les contextes) et, si `expected_answer` est
  fourni, `f1` et `exact_match`.

### Santé

```bash
curl http://localhost:8000/health
curl http://localhost:8000/
```

## Tests

```bash
python -m pytest test -v
```

Les tests API remplacent les embeddings Ollama par des vecteurs déterministes :
ils ne nécessitent ni Ollama ni réseau.

## Notes

- L'interface est **uniquement l'API FastAPI** ; l'ancien code Streamlit a été
  supprimé.
- L'index RAG de l'API (`bdd/rag_index.faiss`) est **distinct** de l'index des
  événements (`bdd/faiss_index.idx`, construit par `indexer.py`).
- `documents/`, `bdd/rag_index.faiss` et `bdd/rag_metadata.pkl` sont générés et
  ignorés par git.
