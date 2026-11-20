import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import numpy as np
import faiss


def chunks_text(texte, chunk_size=800, chunk_overlap=100):
    print('# DEBUT CHUNKS')
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, # Taille de chaque segment
        chunk_overlap=chunk_overlap # Chevauchement entre les segments
        )
        
    # Découpage du texte
    chunks = text_splitter.split_text(texte)

    print('# FIN CHUNK')
    return chunks

# embbeding function

def embed_ollama(texts, batch_size=32) :
    print('# DEBUT EMBEDDING')
    vecteurs = [] # dict

    # boucle
    # Initialisation du modèle local
    model = "embeddinggemma-2:latest"

    for debut in range(0, len(texts), batch_size):
        lot = texts[debut:debut + batch_size]
        # Obtention de l'embedding
        response = ollama.embed(
            model=model,
            input=lot
        )
        
        #print('# RESPONSE : ', response)
        vecteurs.extend(response["embeddings"])
    
    return vecteurs

