
import faiss
import pickle
import numpy as np
from functools import lru_cache

from embedding import embed_ollama


# =====================================
# 1. Chargement FAISS
# =====================================

@lru_cache(maxsize=1)
def charger_index():
    index = faiss.read_index("bdd/faiss_index.idx")
    with open("bdd/metadata.pkl", "rb") as f:
        data = pickle.load(f)

    if not (index.ntotal == len(data["chunks"]) == len(data["sources"])):
        raise ValueError(
            "Index FAISS et chunks désalignés : relance l'ingestion."
        )
    return index, data


# =====================================
# 2. Recherche vectorielle
# =====================================

def rechercher_evenements(question, k=4, score_min=0.76):
    index, data = charger_index()

    query = np.asarray(embed_ollama([question]), dtype="float32")
    faiss.normalize_L2(query)

    scores, indices = index.search(query, min(k, index.ntotal))

    resultats = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or score < score_min:      # ← filtre de pertinence
            continue
        resultats.append({
            "chunk": data["chunks"][idx],
            "evenement": data["metadata"][data["sources"][idx]],
            "score": float(score),
        })
    return resultats
