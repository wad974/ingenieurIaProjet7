import numpy as np
from embedding import chunks_langchain, embed_ollama



# generation des embeddings apres chunks sur chaque document
chunks_data = chunks_langchain(data['results'])
embed = np.array([embed_ollama(chunks_data)])

query_vector = np.array(
    vectors[0],
    dtype="float32"
).reshape(1, -1)

distances, indices = index.search(query_vector, 5)

print("Indices :", indices)
print("Distances :", distances)