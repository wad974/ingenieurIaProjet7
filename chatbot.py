from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from retrieval import rechercher_evenements

llm = ChatOllama(
    model="llama3.1:8b",        # un seul modèle pour tout le projet
    temperature=0.2,
    num_ctx=8192,
    reasoning=False,            # utile si tu passes sur qwen3.5
)

prompt = ChatPromptTemplate.from_template("""
Tu es un assistant culturel de La Réunion.
Tu recommandes uniquement les événements fournis ci-dessous.

Événements disponibles :
{context}

Question : {question}

Consignes :
- Réponds en français, de façon concise.
- Pour chaque recommandation, donne le titre, la date et la ville.
- Explique en une phrase pourquoi l'événement correspond.
- N'invente aucun événement, aucune date, aucun lieu.
- Si aucun événement ne correspond, dis-le explicitement.
""")

chain = prompt | llm | StrOutputParser()

NON_TROUVE = (
    "Je n'ai trouvé aucun événement correspondant à votre demande "
    "dans les événements que je connais."
)


def chatbot(question):
    resultats = rechercher_evenements(question)
    if not resultats:                 # ← pas d'appel LLM sans contexte
        return NON_TROUVE
    context = "\n\n---\n\n".join(r["chunk"] for r in resultats)
    return chain.invoke({"context": context, "question": question})

chatbot('quels sont les evenements?')