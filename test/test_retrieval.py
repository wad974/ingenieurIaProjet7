"""Tests de fumée du chargement des documents et du magasin vectoriel.

Exécution :
    python -m pytest test -v
"""

from __future__ import annotations

import pytest

from utils import config
from utils.data_loader import fiche_evenement
from utils.vector_store import charger_index, decouper_texte


def test_decouper_texte_respecte_la_taille():
    texte = "Phrase de test pour le découpage. " * 200

    chunks = decouper_texte(texte, chunk_size=200, chunk_overlap=20)

    assert chunks
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_fiche_evenement_contient_les_champs_cles():
    fiche = fiche_evenement(
        {
            "title_fr": "Concert",
            "daterange_fr": "12 juillet 2026",
            "location_city": "Saint-Denis",
        }
    )

    assert "Titre : Concert" in fiche
    assert "Date : 12 juillet 2026" in fiche
    assert "Ville : Saint-Denis" in fiche


def test_index_faiss_aligne_avec_les_chunks():
    if not (
        config.FAISS_INDEX_PATH.exists() and config.METADATA_PATH.exists()
    ):
        pytest.skip("Index FAISS absent : lance d'abord python indexer.py")

    index, data = charger_index()

    assert index.ntotal == len(data["chunks"]) == len(data["sources"])
