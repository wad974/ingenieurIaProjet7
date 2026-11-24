"""Tests d'intégration de l'API FastAPI (sans Ollama réel)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import utils.document_store as document_store
import utils.vector_store as vector_store
from utils import config


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Client de test avec un index RAG et des embeddings isolés."""
    index_path = tmp_path / "rag_index.faiss"
    metadata_path = tmp_path / "rag_metadata.pkl"
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    monkeypatch.setattr(document_store, "INDEX_PATH", index_path)
    monkeypatch.setattr(document_store, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(config, "DOCUMENTS_DIR", documents_dir)
    document_store.charger_index_documents.cache_clear()

    def faux_embed(textes, batch_size=None, model=None):
        """Embedding déterministe de dimension 8 (pas d'appel Ollama)."""
        vecteurs = []
        for texte in textes:
            vecteur = np.zeros(8, dtype="float32")
            for position, octet in enumerate(texte.encode("utf-8")):
                vecteur[position % 8] += float(octet % 32)
            if not vecteur.any():
                vecteur[0] = 1.0
            vecteurs.append(vecteur.tolist())
        return vecteurs

    monkeypatch.setattr(vector_store, "embed_texts", faux_embed)

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client

    document_store.charger_index_documents.cache_clear()


def test_health(client):
    reponse = client.get("/health")

    assert reponse.status_code == 200
    assert reponse.json()["status"] == "ok"


def test_flux_prompt_reponse_ingestion(client, monkeypatch):
    import api.routers.prompt as prompt_router

    def faux_repondre(question, top_k=None, score_min=None, mode="auto"):
        contextes = document_store.rechercher_documents(
            question,
            k=top_k,
            score_min=score_min,
        )
        return {
            "answer": "Réponse de test.",
            "contexts": contextes,
            "mode": "RAG",
            "rag_used": True,
            "confidence": 0.9,
            "reason": "test",
        }

    monkeypatch.setattr(prompt_router, "repondre", faux_repondre)

    # Flux 2 : ingestion d'un fichier texte
    reponse = client.post(
        "/ingest",
        files=[
            (
                "files",
                (
                    "test.txt",
                    b"Le concert aura lieu a Saint-Denis le 12 juillet.",
                    "text/plain",
                ),
            )
        ],
        data={"replace": "true"},
    )
    assert reponse.status_code == 200, reponse.text
    corps = reponse.json()
    assert corps["index"]["documents"] == 1
    assert corps["index"]["chunks"] >= 1
    assert corps["sources"][0]["status"] == "ok"

    # Flux 1 : une question -> une réponse JSON immédiate.
    # Métriques désactivées ici pour ne pas appeler Ollama/RAGAS.
    reponse = client.post(
        "/prompt",
        json={
            "question": "concert Saint-Denis",
            "top_k": 3,
            "score_min": -1.0,
            "include_metrics": False,
        },
    )
    assert reponse.status_code == 200, reponse.text
    resultat = reponse.json()
    assert resultat["question"] == "concert Saint-Denis"
    assert resultat["answer"] == "Réponse de test."
    assert resultat["mode"] == "RAG"
    assert resultat["rag_used"] is True
    assert resultat["confidence"] == 0.9
    assert resultat["reason"] == "test"
    assert resultat["contexts"]
    assert resultat["model"]
    assert resultat["metrics"] is None
    assert "elapsed_ms" in resultat
    assert "query_id" not in resultat


def test_fichier_non_supporte(client):
    reponse = client.post(
        "/ingest",
        files=[("files", ("virus.exe", b"abc", "application/octet-stream"))],
    )

    assert reponse.status_code == 400


def test_ingestion_sans_source(client):
    reponse = client.post("/ingest")

    assert reponse.status_code == 400


def test_ingestion_url(client, monkeypatch):
    import api.routers.ingestion as ingestion_router
    from utils.markdown_converter import ConversionResult

    monkeypatch.setattr(
        ingestion_router,
        "depuis_url",
        lambda url: ConversionResult(
            markdown="# Page\n\nLe contenu de la page.",
            title="Page",
        ),
    )

    # `files=""` reproduit le comportement de Swagger quand aucun fichier
    # n'est sélectionné.
    reponse = client.post(
        "/ingest",
        data={
            "urls": "https://example.com/page",
            "files": "",
            "replace": "true",
        },
    )

    assert reponse.status_code == 200, reponse.text
    corps = reponse.json()
    assert corps["index"]["documents"] == 1
    assert corps["sources"][0]["kind"] == "url"
    assert corps["sources"][0]["status"] == "ok"


def test_source_deja_indexee_est_ignoree(client):
    fichier = [("files", ("doc.txt", b"Contenu.", "text/plain"))]

    premier = client.post(
        "/ingest",
        files=fichier,
        data={"replace": "true"},
    )
    assert premier.status_code == 200

    second = client.post("/ingest", files=fichier)
    assert second.status_code == 200
    assert second.json()["sources"][0]["status"] == "skipped"


def test_question_vide_rejetee(client):
    reponse = client.post("/prompt", json={"question": "   "})

    assert reponse.status_code == 422


def test_files_vide_tolere(client):
    """Swagger envoie `files=""` : on doit obtenir une 400 claire, pas une 422."""
    reponse = client.post("/ingest", data={"files": ""})

    assert reponse.status_code == 400
    detail = reponse.json()["detail"]
    assert isinstance(detail, dict)
    assert "Aucun fichier ni URL" in detail["message"]


def test_prompt_mode_direct(client, monkeypatch):
    import api.routers.prompt as prompt_router

    monkeypatch.setattr(
        prompt_router,
        "repondre",
        lambda question, **kwargs: {
            "answer": "Réponse directe.",
            "contexts": [],
            "mode": "DIRECT",
            "rag_used": False,
            "confidence": 0.95,
            "reason": "Salutation",
        },
    )

    reponse = client.post(
        "/prompt",
        json={
            "question": "Bonjour",
            "mode": "direct",
            "include_metrics": False,
        },
    )

    assert reponse.status_code == 200, reponse.text
    corps = reponse.json()
    assert corps["mode"] == "DIRECT"
    assert corps["rag_used"] is False
    assert corps["contexts"] == []
    assert corps["reason"] == "Salutation"


def test_classificateur_rag_ou_direct():
    from utils.query_classifier import classifier_demande

    assert classifier_demande("Bonjour").needs_rag is False
    assert classifier_demande("Quel concert à Saint-Denis ?").needs_rag is True
    assert (
        classifier_demande("Quelle météo ?", utiliser_llm=False).needs_rag
        is False
    )


def test_prompt_avec_metriques(client, monkeypatch):
    import api.routers.prompt as prompt_router

    monkeypatch.setattr(
        prompt_router,
        "repondre",
        lambda question, **kwargs: {
            "answer": "Le concert aura lieu le 12 juillet.",
            "contexts": [
                {
                    "chunk": "Le concert aura lieu le 12 juillet.",
                    "score": 0.9,
                    "document": {
                        "id": "doc1",
                        "title": "concert",
                        "source": "concert.txt",
                        "kind": "file",
                    },
                }
            ],
            "mode": "RAG",
            "rag_used": True,
            "confidence": 0.9,
            "reason": "test",
        },
    )
    monkeypatch.setattr(
        prompt_router,
        "evaluer_reponse",
        lambda **kwargs: {
            "metrics": {
                "hit@k": 1.0,
                "groundedness": 0.9,
                "faithfulness": 0.8,
                "answer_relevancy": 0.7,
                "context_precision": 0.6,
                "context_recall": 0.5,
            },
            "ragas_error": None,
        },
    )

    reponse = client.post(
        "/prompt",
        json={
            "question": "Quand a lieu le concert ?",
            "ground_truth": "Le 12 juillet.",
            "expected_sources": ["concert.txt"],
            "include_metrics": True,
            "include_ragas": True,
        },
    )

    assert reponse.status_code == 200, reponse.text
    corps = reponse.json()
    assert corps["metrics"]["faithfulness"] == 0.8
    assert corps["metrics"]["hit@k"] == 1.0
    assert corps["ragas_error"] is None
    assert corps["metrics_elapsed_ms"] is not None


def test_evaluation_reponse_avec_ragas(monkeypatch):
    import api.services.evaluation as evaluation
    import api.services.ragas_evaluation as ragas_evaluation

    capture: dict = {}

    def faux_ragas(echantillons, metriques=None):
        capture["echantillon"] = echantillons[0]
        capture["metriques"] = metriques
        scores = {
            "faithfulness": 0.8,
            "answer_relevancy": 0.7,
            "context_precision": 0.6,
            "context_recall": 0.5,
        }
        noms = metriques or list(scores)
        return {
            "scores": {nom: scores[nom] for nom in noms},
            "details": [{nom: scores[nom] for nom in noms}],
        }

    monkeypatch.setattr(ragas_evaluation, "evaluer_avec_ragas", faux_ragas)

    contextes = [
        {
            "chunk": "Le concert aura lieu le 12 juillet.",
            "score": 0.9,
            "document": {"source": "concert.txt", "title": "concert"},
        }
    ]
    resultat = evaluation.evaluer_reponse(
        question="Quand a lieu le concert ?",
        answer="Le 12 juillet.",
        mode="RAG",
        contexts=contextes,
        ground_truth="Le 12 juillet.",
        expected_sources=["concert.txt"],
        include_ragas=True,
    )

    assert resultat["ragas_error"] is None
    assert resultat["metrics"]["hit@k"] == 1.0
    assert resultat["metrics"]["groundedness"] == 1.0
    assert resultat["metrics"]["f1"] == 1.0
    assert resultat["metrics"]["exact_match"] == 1.0
    assert resultat["metrics"]["faithfulness"] == 0.8
    assert resultat["metrics"]["context_recall"] == 0.5
    assert capture["metriques"] == [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    assert capture["echantillon"]["contexts"] == [
        "Le concert aura lieu le 12 juillet."
    ]
    assert capture["echantillon"]["ground_truth"] == "Le 12 juillet."


def test_evaluation_reponse_sans_ground_truth(monkeypatch):
    import api.services.evaluation as evaluation
    import api.services.ragas_evaluation as ragas_evaluation

    capture: dict = {}

    def faux_ragas(echantillons, metriques=None):
        capture["metriques"] = metriques
        return {
            "scores": {"faithfulness": 1.0, "answer_relevancy": 0.5},
            "details": [{"faithfulness": 1.0, "answer_relevancy": 0.5}],
        }

    monkeypatch.setattr(ragas_evaluation, "evaluer_avec_ragas", faux_ragas)

    resultat = evaluation.evaluer_reponse(
        question="Quel concert ?",
        answer="Un concert de jazz.",
        mode="RAG",
        contexts=[
            {
                "chunk": "Un concert de jazz aura lieu.",
                "score": 0.8,
                "document": {"source": "doc.txt"},
            }
        ],
        include_ragas=True,
    )

    assert capture["metriques"] == ["faithfulness", "answer_relevancy"]
    assert resultat["metrics"]["faithfulness"] == 1.0
    assert "context_precision" not in resultat["metrics"]


def test_evaluation_reponse_mode_direct(monkeypatch):
    import api.services.evaluation as evaluation
    import api.services.ragas_evaluation as ragas_evaluation

    capture: dict = {}

    def faux_ragas(echantillons, metriques=None):
        capture["metriques"] = metriques
        capture["echantillon"] = echantillons[0]
        return {
            "scores": {"answer_relevancy": 0.6},
            "details": [{"answer_relevancy": 0.6}],
        }

    monkeypatch.setattr(ragas_evaluation, "evaluer_avec_ragas", faux_ragas)

    resultat = evaluation.evaluer_reponse(
        question="Bonjour",
        answer="Bonjour !",
        mode="DIRECT",
        contexts=[],
        include_ragas=True,
    )

    assert capture["metriques"] == ["answer_relevancy"]
    assert capture["echantillon"]["contexts"] == []
    assert resultat["metrics"]["answer_relevancy"] == 0.6


def test_flux_metriques_supprime(client):
    assert client.get("/metrics").status_code == 404
    assert client.post("/metrics", json={}).status_code == 404


def test_url_json_vers_markdown(monkeypatch):
    import utils.markdown_converter as convertisseur

    class FausseReponse:
        status_code = 200
        headers = {"content-type": "application/json; charset=utf-8"}
        content = b"{}"
        text = "{}"

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "title_fr": "Concert de jazz",
                        "daterange_fr": "12 juillet 2026",
                        "location_city": "Saint-Denis",
                    }
                ]
            }

    monkeypatch.setattr(
        convertisseur.requests,
        "get",
        lambda *args, **kwargs: FausseReponse(),
    )

    resultat = convertisseur.depuis_url("https://exemple.com/api/records")

    assert "Concert de jazz" in resultat.markdown
    assert "Saint-Denis" in resultat.markdown
