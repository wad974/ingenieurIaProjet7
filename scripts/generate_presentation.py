"""Génère la présentation PowerPoint de soutenance (13 slides).

Usage :
    ./env/bin/python scripts/generate_presentation.py

Produit : presentation/Projet7_Soutenance_RAG.pptx
Les chiffres (résultats, temps de réponse) s'appuient sur les tests et les
runs de démo réels ; adaptez PRESENTATION_DATA si besoin.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# --- Données à jour ----------------------------------------------------------
PRESENTATION_DATA = {
    "event_count": 500,
    "chunks": 83,
    "dim": 768,
    "tests": 19,
    "metrics_demo": {
        "hit@k": "1.0",
        "mrr": "1.0",
        "faithfulness (RAGAS)": "0.75",
        "answer_relevancy (RAGAS)": "0.79",
        "context_precision (RAGAS)": "1.0",
    },
}

BLEU = RGBColor(0x1F, 0x4E, 0x79)
GRIS = RGBColor(0x59, 0x59, 0x59)
BLANC = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT = RGBColor(0x2E, 0x74, 0xB5)
VERT = RGBColor(0x4E, 0x79, 0x1F)
BRUN = RGBColor(0x7F, 0x3F, 0x00)

OUT = Path(__file__).resolve().parent.parent / "presentation"
OUT.mkdir(exist_ok=True)
FICHIER = OUT / "Projet7_Soutenance_RAG.pptx"


def puces(slide, elements: list[str], placeholder_index: int = 1) -> None:
    corps = slide.placeholders[placeholder_index].text_frame
    corps.clear()
    for i, element in enumerate(elements):
        p = corps.paragraphs[0] if i == 0 else corps.add_paragraph()
        p.text = element
        p.level = 0
        p.font.size = Pt(16)


def encadre(slide, gauche: float, haut: float, largeur: float, hauteur: float,
            titre: str, lignes: list[str],
            couleur: RGBColor = ACCENT) -> None:
    """Boîte de texte encadrée pour les schémas simples."""
    boite = slide.shapes.add_textbox(Inches(gauche), Inches(haut),
                                     Inches(largeur), Inches(hauteur))
    remplissage = boite.fill
    remplissage.solid()
    remplissage.fore_color.rgb = couleur
    tf = boite.text_frame
    tf.word_wrap = True
    tf.text = titre
    p = tf.paragraphs[0]
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLANC
    p.alignment = PP_ALIGN.CENTER
    for ligne in lignes:
        p = tf.add_paragraph()
        p.text = ligne
        p.font.size = Pt(11)
        p.font.color.rgb = BLANC
        p.alignment = PP_ALIGN.CENTER


def nouvelle_slide(prs: Presentation, titre: str):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = titre
    return slide


def main() -> None:
    prs = Presentation()

    # --- Slide 1 : titre -----------------------------------------------------
    titre = prs.slides.add_slide(prs.slide_layouts[0])
    titre.shapes.title.text = "Assistant documentaire RAG"
    titre.placeholders[1].text = (
        "Ollama + FAISS — pipeline local, API FastAPI conteneurisée\n"
        "Projet 7 — IngenieurIA | 2026"
    )

    # --- Slide 2 : problème & contexte ---------------------------------------
    s = nouvelle_slide(prs, "Problème & contexte")
    puces(s, [
        "Les événements culturels sont dispersés : portails open data, "
        "sites de salles, brochures PDF — difficiles à interroger.",
        "Une recherche par mots-clés ne comprend pas les questions en "
        "langage naturel (« Quels concerts de jazz à Paris ce week-end ? »).",
        "Un LLM seul répond « de mémoire » : risques d'invention "
        "(hallucination) et aucune connaissance des données locales.",
        "Objectif : une API conversationnelle qui répond en s'appuyant "
        "exclusivement sur une base documentaire vérifiable.",
    ])

    # --- Slide 3 : le RAG expliqué simplement ---------------------------------
    s = nouvelle_slide(prs, "Le RAG en 4 étapes (pour les non-techniciens)")
    puces(s, [
        "1. Indexer : les documents sont découpés en extraits et "
        "transformés en vecteurs (empreintes numériques du sens).",
        "2. Chercher : à chaque question, les extraits les plus proches du "
        "sens de la question sont retrouvés par similarité.",
        "3. Répondre : le LLM rédige une réponse à partir de ces extraits "
        "seulement — il ne répond plus « de mémoire ».",
        "4. Évaluer : chaque réponse est accompagnée de métriques (qu'a-t-il "
        "trouvé ? fidèle ? pertinent ?) — la traçabilité est intégrée.",
    ])
    encadre(s, 4.6, 4.0, 4.6, 1.4, "Bénéfice métier",
            ["Réponses concrètes et sourcées",
             "Mise à jour = ré-ingestion, pas de ré-entraînement"])

    # --- Slide 4 : solution proposée ------------------------------------------
    s = nouvelle_slide(prs, "Solution : une API RAG à 2 flux")
    puces(s, [
        "Flux 1 — POST /prompt : question → classification RAG/DIRECT → "
        "recherche FAISS → génération Ollama → réponse + métriques en un "
        "seul appel JSON.",
        "Flux 2 — POST /ingest : fichiers (PDF, DOCX, TXT, MD, HTML) et "
        "URLs → Markdown → chunks → embeddings → index FAISS mis à jour.",
        "Documentation interactive Swagger UI (/docs) pour la démo et les "
        "parties prenantes.",
        "Lancement local simplifié : docker compose up --build -d puis "
        "curl http://localhost:8000/health.",
    ])

    # --- Slide 5 : architecture -------------------------------------------------
    s = nouvelle_slide(prs, "Architecture technique")
    encadre(s, 0.6, 1.6, 2.4, 1.0, "Client",
            ["curl / Swagger", "Postman, navigateur"], GRIS)
    encadre(s, 3.4, 1.6, 2.6, 1.0, "API FastAPI",
            ["/prompt — /ingest", "classifier + métriques"], ACCENT)
    encadre(s, 6.4, 1.6, 2.6, 1.0, "FAISS",
            ["IndexFlatIP", "cosinus (L2 norm)"], VERT)
    encadre(s, 3.4, 3.0, 2.6, 1.0, "Ollama",
            ["qwen3.5 (LLM)", "embeddinggemma-2 (768d)"], BRUN)
    encadre(s, 0.6, 4.4, 2.4, 1.0, "Sources",
            ["PDF/DOCX/HTML/URL", "OpenAgenda"], GRIS)
    encadre(s, 3.4, 4.4, 2.6, 1.0, " ingestion",
            ["Markdown → chunks", "embeddings → index"], GRIS)
    puces(s, [
        "Tout tourne en local (conteneur) : aucune donnée ne sort, aucune "
        "clé API, coût marginal nul.",
    ])

    # --- Slide 6 : conteneurisation ----------------------------------------------
    s = nouvelle_slide(prs, "Conteneurisation Docker")
    puces(s, [
        "docker-compose : 2 services — `ollama` (LLM + embeddings) et "
        "`api` (FastAPI), réseau interne privé, un seul port exposé.",
        "Image API : python:3.14-slim, dépendances verrouillées "
        "(requirements.txt), entrypoint qui prépare les dossiers puis "
        "uvicorn.",
        "Volumes : index FAISS + documents + logs montés depuis l'hôte → "
        "état persistant ; modèles dans le volume Docker `ollama_models`.",
        "Service `ollama-init` : télécharge les modèles une seule fois ; "
        "les démarrages suivants sont hors-ligne (démo sans réseau stable).",
        "Variante GPU prête : NVIDIA Container Toolkit → décommenter le "
        "bloc deploy du service ollama.",
    ])

    # --- Slide 7 : données & modèles ---------------------------------------------
    s = nouvelle_slide(prs, "Données & modèles choisis")
    puces(s, [
        "Données : jeu public OpenAgenda / OpenDataSoft — événements "
        f"culturels ({PRESENTATION_DATA['event_count']} événements "
        "indexés), complété par documents PDF/DOCX/URL via /ingest.",
        f"Corpus indexé : {PRESENTATION_DATA['chunks']} chunks, vecteurs de "
        f"dimension {PRESENTATION_DATA['dim']}, similarité cosinus "
        "(IndexFlatIP, normalisation L2).",
        "Embeddings : embeddinggemma-2 (Google, 768d) via Ollama — local, "
        "gratuit, reproductible.",
        "LLM : qwen3.5 (Ollama) — aucune donnée n'est envoyée en ligne, "
        "coût marginal nul.",
        "Découpage : RecursiveCharacterTextSplitter — 800 caractères, "
        "chevauchement 100.",
    ])

    # --- Slide 8 : routage RAG vs DIRECT -----------------------------------------
    s = nouvelle_slide(prs, "Routage intelligent : RAG vs DIRECT")
    puces(s, [
        "Un classificateur décide si la question nécessite la base "
        "documentaire (inspiré du cours OpenClassrooms).",
        "Heuristiques instantanées : mots-clés métier (concert, festival, "
        "dates, lieux…), salutations, questions de capacité.",
        "Cas ambigus : un appel LLM local décide — coût limité aux "
        "questions vraiment ambiguës.",
        "En mode DIRECT (question générale), le LLM répond sans retrieval ; "
        "en mode RAG, la réponse cite les documents utilisés.",
        "Un mode `rag` ou `direct` peut être forcé pour les tests et la "
        "démo.",
    ])

    # --- Slide 9 : évaluation -----------------------------------------------------
    s = nouvelle_slide(prs, "Évaluation du système")
    puces(s, [
        "Retrieval — hit@k : trouve le bon document ? mrr : à quelle "
        "position ? precision@k / recall@k.",
        "Génération — groundedness : la réponse s'appuie-t-elle sur les "
        "extraits ? f1 / exact_match vs réponse attendue.",
        "RAGAS — faithfulness (fidélité aux sources), answer_relevancy "
        "(répond à la question ?), context_precision / context_recall.",
        "Les métriques sont renvoyées avec chaque réponse /prompt (JSON) : "
        "l'évaluation fait partie du produit.",
    ])
    encadre(s, 4.6, 4.4, 4.6, 1.4, "Valeurs observées (démo)",
            [f"{k} = {v}"
             for k, v in PRESENTATION_DATA["metrics_demo"].items()])

    # --- Slide 10 : qualité & tests ------------------------------------------------
    s = nouvelle_slide(prs, "Qualité : tests automatisés")
    puces(s, [
        f"{PRESENTATION_DATA['tests']} tests unitaires et d'intégration "
        "(pytest) : retrieval, API, ingestion, métriques.",
        "Les tests API neutralisent Ollama (vecteurs déterministes) et "
        "mockent RAGAS : ils tournent en ~2 s, sans réseau ni GPU.",
        "Chaîne validée de bout en bout en conteneur : santé → ingestion → "
        "questions RAG/DIRECT → métriques.",
        "Journalisation (console + fichier rotatif) et /health pour le "
        "suivi opérationnel.",
    ])

    # --- Slide 11 : démonstration --------------------------------------------------
    s = nouvelle_slide(prs, "Démonstration (live)")
    puces(s, [
        "Scénario métier : « Quels concerts de jazz sont organisés ? » → "
        "réponse RAG sourcée, en une requête POST.",
        "Scénario corpus : « Quels spectacles gratuits ce week-end et "
        "où ? » → réponse construite à partir des chunks événements.",
        "Scénario hors corpus : « Programme du Festival de Cannes 2030 ? » "
        "→ l'API reconnaît l'absence au lieu d'inventer.",
        "Outils : Swagger UI + scripts/demo.sh — tout tourne en local, "
        "y compris le LLM.",
    ])

    # --- Slide 12 : perspectives ---------------------------------------------------
    s = nouvelle_slide(prs, "Perspectives d'amélioration")
    puces(s, [
        "Hybride FAISS + recherche plein-texte (BM25) pour les requêtes par "
        "nom exact (dates, titres d'événements).",
        "Reranking des contextes (cross-encoder) pour remonter la précision "
        "top-1 sans changer le LLM.",
        "Interface métier légère (Streamlit/Gradio) branchée sur l'API, "
        "pour les équipes non techniques.",
        "Suivi continu : tableau RAGAS sur un jeu de questions type, "
        "exécuté à chaque ré-ingestion.",
        "Déploiement : registre d'images + orchestration (compose ou K8s) ; "
        "montée de charge via plusieurs répliques uvicorn.",
    ])

    # --- Slide 13 : merci ----------------------------------------------------------
    s = nouvelle_slide(prs, "Merci — vos questions ?")
    puces(s, [
        "Réponses disponibles : architecture, métriques, conteneurs, "
        "coûts, limites du RAG.",
        "Le projet est reproductible : docker compose up --build -d.",
    ])

    prs.save(FICHIER)
    print(f"Présentation générée : {FICHIER}")


if __name__ == "__main__":
    main()
