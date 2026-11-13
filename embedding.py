import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

def chunks_langchain(files):
    print('# DEBUT CHUNKS')
    
    # convert json text
    text = json.dumps(files, ensure_ascii=False, indent=2)
    # Initialisation du splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Taille de chaque segment
        chunk_overlap=200 # Chevauchement entre les segments
        )
        
    # Découpage du texte
    chunks = text_splitter.split_text(text)

    print('# FIN CHUNKS')
    return chunks

# embbeding function

def embed_ollama(documents) :
    print('# DEBUT EMBEDDING')

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
