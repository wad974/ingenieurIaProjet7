import requests

url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

params = {
    "where": """location_postalcode LIKE '974%'
        AND firstdate_begin >= '2010-12-29T00:00:00'
        AND lastdate_end < '2026-12-30T00:00:00'
    """,
    "order_by": "firstdate_begin ASC",
    "limit": 100
}

response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()

print("Nombre de résultats :", data["total_count"])

documents = []
for event in data["results"]:
    # filtre si none
    if event.get("title_fr") is not None:
        documents.append(event)
        print('*'*50)
        print(event)
        
# embedding
from embedding import embed_ollama, chunks_langchain
import numpy as np
import faiss

# generation des embeddings apres chunks sur chaque document
chunks_data = chunks_langchain(data['results'])
embed = np.array([embed_ollama(chunks_data)])

#Initialisation de l'index faiss
dimension = embed.shape[1]
index = faiss.IndexFlatL2(dimension)

#ajout embedding dans index
index.add(embed)

#sauvegarde index sur le disque bdd/ 
faiss.write_index(index, "bdd/faiss_index.idx")
