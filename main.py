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

tous_les_embeddings = []
for row in documents:
    
    chunks_data = chunks_langchain([row])
    embedding = embed_ollama(chunks_data)
    
    tous_les_embeddings.append(embedding)

# conversion numpy
embedding_for_faiss = np.array(tous_les_embeddings, dtype='float32')

#Initialisation de l'index faiss
dimension = embedding_for_faiss.shape[1]
index = faiss.IndexFlatL2(dimension)

#ajout embedding dans index
index.add(embedding_for_faiss)

#sauvegarde index sur le disque bdd/ 
faiss.write_index(index, "bdd/faiss_index.idx")

# verification que tous les evenements sont indexés
print("Nombre total d'événements :", len(documents))#31
print("Nombre de vecteurs :", len(embedding_for_faiss) ,' - ', dimension) # 31 - 768
print("Nombre de vecteurs dans FAISS :", index.ntotal) # 31