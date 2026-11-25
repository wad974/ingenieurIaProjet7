# API RAG — Ollama + FAISS

API FastAPI à **2 flux** pour un assistant documentaire RAG :

| Flux | Endpoint | Rôle |
|------|----------|------|
| 1. Prompt + Réponse + Métriques | `POST /prompt` | Classe la question **RAG ou DIRECT**, cherche dans FAISS si nécessaire, génère la réponse Ollama et renvoie **la réponse avec ses métriques** (retrieval, génération, RAGAS) |
| 2. Ingestion | `POST /ingest` | Fichiers et/ou URLs → **Markdown** → chunks → embeddings → index FAISS sauvegardé |

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
│   │   ├── prompt.py           # Flux 1 : POST /prompt (réponse + métriques)
│   │   └── ingestion.py        # Flux 2 : POST /ingest
│   └── services/
│       ├── evaluation.py       # Métriques intégrées à la réponse /prompt
│       └── ragas_evaluation.py # Les 4 métriques RAGAS
├── utils/
│   ├── config.py               # Configuration (.env), chemins, modèles
│   ├── markdown_converter.py   # PDF/DOCX/TXT/MD/HTML/URL -> Markdown
│   ├── document_store.py       # Index FAISS des documents (recherche, ajout)
│   ├── document_ingestion.py   # Orchestration Markdown -> chunks -> index
│   ├── query_classifier.py     # Routage RAG vs DIRECT
│   ├── rag.py                  # Prompts RAG/DIRECT + génération Ollama
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

## Exécution en conteneur Docker (recommandé pour la démo)

Deux conteneurs sur un réseau interne : **Ollama** (LLM + embeddings, modèles
mis en cache dans un volume Docker) et **l'API FastAPI**. Le port hôte est
configurable via `API_PORT` (8000 par défaut).

```bash
docker compose up --build -d       # build + démarrage complet
curl http://localhost:8000/health  # vérification
docker compose logs -f api         # suivre les journaux
```

Le premier démarrage télécharge les modèles (`OLLAMA_MODEL` +
`embeddinggemma-2`) dans le volume `ollama_models` — comptez quelques
minutes. Ensuite, la stack démarre en secondes et **fonctionne hors-ligne** :
seul le port de l'API est exposé. L'index FAISS et les documents ingérés sont
montés depuis `./bdd` et `./documents` : l'état créé côté hôte persiste.

Si le port 8000 est déjà occupé (par exemple par un uvicorn local) :

```bash
API_PORT=8001 docker compose up --build -d   # autre port hôte
```

### Construire l'index vectoriel

Construit `bdd/faiss_index.idx` (index des événements OpenAgenda) à
l'intérieur du conteneur API ; la connexion Internet n'est nécessaire que
pour charger les événements et, au premier lancement, les modèles :

```bash
docker compose up -d
./scripts/build_index.sh          # 500 événements (valeur de .env)
./scripts/build_index.sh 200      # nombre d'événements personnalisé
```

> L'index RAG de l'API (`bdd/rag_index.faiss`) se construit différemment :
> via `POST /ingest`, ou simplement avec le script de démo ci-dessous.

### Démo bout-en-bout

Script complet : santé → ingestion du corpus → questions de démonstration
(scénarios RAG et DIRECT). `--quick` désactive les métriques pour des
réponses immédiates :

```bash
docker compose up -d --wait
./scripts/demo.sh --quick         # démo live fluide
./scripts/demo.sh                 # avec métriques et RAGAS
```

Swagger UI pour la présentation et les tests d'endpoints :
`http://localhost:8000/docs`.

### Variante GPU (facultatif)

Le compose tourne par défaut sur CPU. Avec le NVIDIA Container Toolkit
installé (`nvidia-container-toolkit`), décommentez le bloc `deploy` du service
`ollama` dans `docker-compose.yml` et relancez : l'inférence utilisera le GPU
et les réponses seront bien plus rapides.

## Utilisation

### Flux 1 — poser une question et obtenir la réponse JSON

Le corps de la requête est un `BaseModel` `PromptRequest` avec un champ
`question` (visible comme input dans `/docs`) :

```bash
curl -X POST http://localhost:8000/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quel concert de jazz à Saint-Denis ?",
    "mode": "auto",
    "ground_truth": "Un concert de jazz le 12 juillet 2026 à Saint-Denis.",
    "expected_sources": ["programme.pdf"],
    "include_metrics": true,
    "include_ragas": true
  }'
```

Réponse JSON complète (`PromptResponse`) :

```json
{
  "question": "Quel concert de jazz à Saint-Denis ?",
  "answer": "Le concert de jazz aura lieu le 12 juillet 2026 à Saint-Denis, au parc de la Source.",
  "mode": "RAG",
  "rag_used": true,
  "confidence": 0.9,
  "reason": "Mots-clés liés aux événements/documents : concert, saint-denis",
  "contexts": [
    {
      "chunk": "Le 12 juillet 2026, un concert de jazz…",
      "score": 0.804,
      "document": {"id": "…", "title": "programme", "source": "programme.pdf", "kind": "file"}
    }
  ],
  "model": "qwen3.5:2b",
  "elapsed_ms": 71433.7,
  "metrics": {
    "hit@k": 1.0,
    "precision@k": 1.0,
    "recall@k": 1.0,
    "mrr": 1.0,
    "groundedness": 0.5278,
    "f1": 0.3951,
    "exact_match": 0.0,
    "faithfulness": 0.75,
    "answer_relevancy": 0.7901,
    "context_precision": 1.0,
    "context_recall": 1.0
  },
  "ragas_error": null,
  "metrics_elapsed_ms": 66821.4
}
```

#### Mode RAG ou DIRECT

Comme dans le cours OpenClassrooms, un classificateur décide si la question
nécessite le RAG :

- **RAG** : question sur les événements, dates, lieux, programmes ou documents
  indexés. Détection par mots-clés, puis LLM local pour les cas ambigus.
- **DIRECT** : salutation, question générale, météo, question sur les capacités
  de l'assistant, etc.

> Le classificateur n'appelle le LLM local que pour les questions ambiguës ;
> les salutations et les mots-clés métier sont traités instantanément.

Le champ `mode` permet de forcer le comportement :

```bash
# Forcer le RAG
curl -X POST http://localhost:8000/prompt \
  -H "Content-Type: application/json" \
  -d '{"question": "Quel concert à Saint-Denis ?", "mode": "rag"}'

# Forcer le LLM seul, sans consultation de l'index
curl -X POST http://localhost:8000/prompt \
  -H "Content-Type: application/json" \
  -d '{"question": "Quel concert à Saint-Denis ?", "mode": "direct"}'
```

En mode `DIRECT`, `contexts` est vide et `rag_used` vaut `false`. En mode
`RAG`, si aucun chunk n'est trouvé, la réponse l'indique explicitement.

> Le mode `DIRECT` utilise les connaissances générales du modèle : il peut
> inventer des informations locales. C'est précisément ce que le RAG évite.

#### Métriques intégrées à la réponse

`include_metrics=true` (défaut) ajoute un objet `metrics` à la réponse :

- **Retrieval** : `hit@k`, `precision@k`, `recall@k`, `mrr`, calculés si
  `expected_sources` ou `expected_snippets` sont fournis.
- **Génération** : `groundedness` (dès qu'il y a des contextes), `f1` et
  `exact_match` si `ground_truth` est fourni.
- **RAGAS** (`include_ragas=true`, défaut) :
  - mode `DIRECT` : `answer_relevancy` ;
  - mode `RAG` : `faithfulness` + `answer_relevancy` ;
  - mode `RAG` avec `ground_truth` : ajoute `context_precision` et
    `context_recall`.

Notes :

- `ground_truth` est nécessaire pour `context_precision`, `context_recall`,
  `f1` et `exact_match`.
- RAGAS appelle Ollama plusieurs fois : comptez environ **1 minute** par
  prompt. `metrics_elapsed_ms` indique le temps passé dans les métriques.
- Mettre `"include_metrics": false` (ou `"include_ragas": false`) pour une
  réponse plus rapide.
- En cas d'échec RAGAS, la réponse reste disponible et `ragas_error` contient
  le détail.

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

### Santé

```bash
curl http://localhost:8000/health
curl http://localhost:8000/
```

## Adaptation du cours OpenClassrooms

- `utils/query_classifier.py` reprend le `QueryClassifier` du dépôt du cours
  (heuristiques + classification LLM) pour router **RAG vs DIRECT**.
- `api/services/ragas_evaluation.py` reprend le notebook `P2C4_RAGASS.ipynb`
  avec `faithfulness`, `answer_relevancy`, `context_precision` et
  `context_recall`.
- Les modèles Mistral du cours sont remplacés par **Ollama**
  (`ChatOllama` + `OllamaEmbeddings`), tout en conservant l'architecture API
  du projet.
- Les métriques RAGAS sont intégrées à la réponse de `POST /prompt` ; il n'y a
  plus de flux `/metrics` séparé.

## Tests

```bash
python -m pytest test -v
```

Les tests API remplacent les embeddings Ollama par des vecteurs déterministes
et mockent RAGAS : ils ne nécessitent ni Ollama ni réseau.

## Notes

- L'interface est **uniquement l'API FastAPI** ; l'ancien code Streamlit a été
  supprimé.
- Le flux `/metrics` séparé a été supprimé : les métriques sont directement
  renvoyées par `POST /prompt`.
- L'index RAG de l'API (`bdd/rag_index.faiss`) est **distinct** de l'index des
  événements (`bdd/faiss_index.idx`, construit par `indexer.py`).
- `documents/`, `bdd/rag_index.faiss` et `bdd/rag_metadata.pkl` sont générés et
  ignorés par git.
