import requests
from embedding import chunks_text, embed_ollama
import os
import faiss
import pickle
import numpy as np

URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
DATE_MIN = "2026-01-01T00:00:00"

CHAMPS = (
    "uid,title_fr,description_fr,longdescription_fr,conditions_fr,"
    "daterange_fr,firstdate_begin,lastdate_end,location_name,location_city,"
    "location_address,location_postalcode,canonicalurl,category"
)


def recuperer_evenements(max_evenements=500):
    resultats, offset = [], 0
    while len(resultats) < max_evenements:
        params = {
            "where": f"location_region = 'La Réunion' AND firstdate_begin >= '{DATE_MIN}'",
            "order_by": "firstdate_begin ASC",
            "limit": 100,
            "offset": offset,
            "select": CHAMPS,
        }
        response = requests.get(URL, params=params, timeout=60)
        response.raise_for_status()
        page = response.json()["results"]
        if not page:
            break
        resultats.extend(page)
        offset += 100
    return resultats[:max_evenements]

def fiche_evenement(e):
    description = e.get("description_fr") or e.get("longdescription_fr") or ""
    return "\n".join([
        f"Titre : {e.get('title_fr') or 'Sans titre'}",
        f"Date : {e.get('daterange_fr') or e.get('firstdate_begin') or 'Non précisée'}",
        f"Ville : {e.get('location_city') or 'Non précisée'}",
        f"Lieu : {e.get('location_name') or 'Non précisé'}",
        f"Catégorie : {e.get('category') or 'Non précisée'}",
        f"Conditions : {e.get('conditions_fr') or 'Non précisées'}",
        f"Description : {description[:1200]}",
    ])

def construire_index():
    evenements = [e for e in recuperer_evenements() if e.get("title_fr")]

    # dédoublonne les enregistrements identiques (même uid + même date)
    uniques, vus = [], set()
    for e in evenements:
        cle = (e.get("uid"), e.get("firstdate_begin"))
        if cle not in vus:
            vus.add(cle)
            uniques.append(e)
    evenements = uniques

    chunks, sources = [], []
    for i, evenement in enumerate(evenements):
        for morceau in chunks_text(fiche_evenement(evenement)):
            chunks.append(morceau)
            sources.append(i)

    vecteurs = np.asarray(embed_ollama(chunks), dtype="float32")
    faiss.normalize_L2(vecteurs)                 # cosinus via produit scalaire

    index = faiss.IndexFlatIP(vecteurs.shape[1])
    index.add(vecteurs)

    # le garde-fou qui manquait
    assert index.ntotal == len(chunks) == len(sources)

    os.makedirs("bdd", exist_ok=True)
    faiss.write_index(index, "bdd/faiss_index.idx")
    with open("bdd/metadata.pkl", "wb") as f:
        pickle.dump(
            {"chunks": chunks, "sources": sources, "metadata": evenements},
            f,
        )

    print(f"{len(evenements)} événements → {len(chunks)} chunks → {index.ntotal} vecteurs")


if __name__ == "__main__":
    construire_index()