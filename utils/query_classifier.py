"""Classification d'une question : réponse RAG ou réponse directe du LLM.

Reprend l'approche du cours OpenClassrooms : heuristiques (salutations,
mots-clés métier), puis classification par le LLM local pour les cas ambigus.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

from langchain_ollama import ChatOllama

from . import config

logger = logging.getLogger(__name__)

# Mots-clés liés aux événements et à La Réunion : ils déclenchent le RAG.
MOTS_CLES_DOMAINE = (
    "la réunion",
    "reunion",
    "événement",
    "evenement",
    "concert",
    "exposition",
    "atelier",
    "festival",
    "spectacle",
    "marché",
    "marche",
    "visite",
    "culture",
    "sortie",
    "conférence",
    "conference",
    "animation",
    "loisir",
    "programme",
    "agenda",
    "document",
    "saint-denis",
    "saint-paul",
    "saint-pierre",
    "le port",
    "la possession",
    "saint-andré",
    "saint-andre",
    "saint-benoît",
    "saint-benoit",
    "saint-leu",
    "saint-gilles",
    "cilaos",
    "salazie",
    "l'étang-salé",
    "etang-salé",
)

# Questions générales qui ne nécessitent pas de RAG.
PATTERNS_GENERAUX = (
    r"^(bonjour|salut|hello|coucou|hey|bonsoir)[\s\.,!]*$",
    r"^(merci|thanks|thank you|je te remercie)[\s\.,!]*$",
    r"^(comment ça va|ça va|comment vas-tu|comment allez-vous)[\s\.,!?]*$",
    r"^(au revoir|bye|à bientôt|à plus tard|à la prochaine)[\s\.,!]*$",
    r"^(qui es[- ]tu|qu'es[- ]tu|que fais[- ]tu|comment fonctionnes[- ]tu|tu es quoi)[\s\?]*$",
    r"^(aide|help|sos|besoin d'aide)[\s\.,!?]*$",
    r"^(quel temps|quelle météo|quelle meteo|météo|meteo)\b.*",
    r"^(quelle heure|quel jour)\b.*",
)

_llm: ChatOllama | None = None
_lock = threading.Lock()


@dataclass(frozen=True)
class Classification:
    """Décision de routage : RAG ou réponse directe."""

    needs_rag: bool
    confidence: float
    reason: str

    @property
    def mode(self) -> str:
        return "RAG" if self.needs_rag else "DIRECT"


def _obtenir_llm() -> ChatOllama:
    """Petit LLM dédié à la classification (lazy et thread-safe)."""
    global _llm
    if _llm is None:
        with _lock:
            if _llm is None:
                _llm = ChatOllama(
                    model=config.OLLAMA_MODEL,
                    temperature=0,
                    num_ctx=config.OLLAMA_NUM_CTX,
                    num_predict=8,
                    base_url=config.OLLAMA_BASE_URL,
                    reasoning=False,
                )
    return _llm


def classifier_demande(
    question: str,
    utiliser_llm: bool = True,
) -> Classification:
    """Détermine si la question doit utiliser le RAG ou le LLM seul."""
    question = question.strip()
    question_lower = question.lower()

    # 1. Salutations et questions générales.
    for pattern in PATTERNS_GENERAUX:
        if re.match(pattern, question_lower):
            return Classification(
                False,
                0.95,
                "Question générale ou salutation",
            )

    # 2. Mots-clés liés aux événements / documents indexés.
    mots_trouves = [
        mot for mot in MOTS_CLES_DOMAINE if mot in question_lower
    ]
    if mots_trouves:
        return Classification(
            True,
            0.9,
            "Mots-clés liés aux événements/documents : "
            + ", ".join(mots_trouves[:5]),
        )

    # 3. Cas ambigu : classification par le LLM local.
    if utiliser_llm:
        try:
            return _classifier_avec_llm(question)
        except Exception:
            logger.exception(
                "Échec de la classification LLM, repli sur l'heuristique"
            )

    # 4. Repli heuristique.
    if len(question.split()) > 5:
        return Classification(
            True,
            0.6,
            "Question longue sans mot-clé : RAG par précaution",
        )
    return Classification(False, 0.5, "Aucun critère spécifique détecté")


def _classifier_avec_llm(question: str) -> Classification:
    prompt = f"""Tu classes une question pour un assistant d'événements
culturels à La Réunion.

Réponds UNIQUEMENT par RAG ou DIRECT, sans explication.

RAG : la question demande des informations sur des événements, dates, lieux,
      programmes ou documents indexés.
DIRECT : salutation, question générale, météo, question sur tes capacités,
         ou sujet sans lien avec les événements indexés.

Exemples :
Question : Bonjour, comment ça va ? -> DIRECT
Question : Quels sont les événements à Saint-Denis ? -> RAG
Question : Quel concert de jazz à Saint-Denis ? -> RAG
Question : Quel temps fait-il aujourd'hui ? -> DIRECT
Question : Qu'est-ce que l'intelligence artificielle ? -> DIRECT

Question : {question}
Réponse :"""

    reponse = _obtenir_llm().invoke(prompt).content.strip().upper()
    logger.info("Classification LLM pour %r : %r", question, reponse)

    if reponse.startswith("RAG"):
        return Classification(True, 0.85, "Classification LLM : RAG")
    if reponse.startswith("DIRECT"):
        return Classification(False, 0.85, "Classification LLM : DIRECT")

    return Classification(
        True,
        0.6,
        f"Réponse ambiguë ({reponse[:30]}), RAG par précaution",
    )
