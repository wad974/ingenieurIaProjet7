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


def test_flux_prompt_reponse_ingestion_metriques(client, monkeypatch):
    import api.routers.prompt as prompt_router

    monkeypatch.setattr(
        prompt_router,
        "generer_reponse",
        lambda question, contextes: "Réponse de test.",
    )

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

    # État de l'index
    etat = client.get("/metrics")
    assert etat.status_code == 200
    assert etat.json()["documents"] == 1

    # Flux 1 : une question -> une réponse JSON immédiate
    reponse = client.post(
        "/prompt",
        json={"question": "concert Saint-Denis", "top_k": 3, "score_min": -1.0},
    )
    assert reponse.status_code == 200, reponse.text
    resultat = reponse.json()
    assert resultat["question"] == "concert Saint-Denis"
    assert resultat["answer"] == "Réponse de test."
    assert resultat["contexts"]
    assert resultat["model"]
    assert "elapsed_ms" in resultat
    assert "query_id" not in resultat

    # Flux 3 : métriques de retrieval
    reponse = client.post(
        "/metrics",
        json={
            "cases": [
                {
                    "question": "concert",
                    "expected_sources": ["test.txt"],
                }
            ],
            "top_k": 3,
            "score_min": -1.0,
        },
    )
    assert reponse.status_code == 200, reponse.text
    metriques = reponse.json()
    assert metriques["aggregate"]["hit@k"] == 1.0
    assert metriques["aggregate"]["n_retrieval_evalues"] == 1


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
