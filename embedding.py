import ollama

# embbeding function

def embed_ollama(documents) :

    # Initialisation du modèle local
    model = "embeddinggemma-2:latest "

    # Texte à vectoriser
    text = documents

    # Obtention de l'embedding
    response = ollama.embed(
        model=model,
        input=text
    )

    embedding = response["embeddings"][0]

    print("Embedding :")
    print(embedding)

    print("\nDimension :", len(embedding))
    
    return embedding
