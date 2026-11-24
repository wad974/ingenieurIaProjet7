"""Préparation de la base de connaissances (index vectoriel FAISS).

Usage :
    python indexer.py
    python indexer.py --max-evenements 200
"""

from __future__ import annotations

import argparse
import logging

from utils import config
from utils.data_loader import charger_evenements, fiche_evenement
from utils.logging_config import setup_logging
from utils.vector_store import construire_index, decouper_texte

logger = logging.getLogger(__name__)


def construire_base(max_evenements: int | None = None) -> tuple[int, int]:
    """Récupère les événements, les découpe et reconstruit l'index vectoriel."""
    evenements = charger_evenements(max_evenements=max_evenements)

    chunks: list[str] = []
    sources: list[int] = []
    for numero, evenement in enumerate(evenements):
        for morceau in decouper_texte(fiche_evenement(evenement)):
            chunks.append(morceau)
            sources.append(numero)

    if not chunks:
        raise RuntimeError("Aucun chunk produit : vérifie la source de données")

    logger.info("%d événements -> %d chunks", len(evenements), len(chunks))
    index = construire_index(chunks, sources, evenements)
    return len(evenements), index.ntotal


def main(argv: list[str] | None = None) -> None:
    """Point d'entrée en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Construit l'index vectoriel FAISS des événements."
    )
    parser.add_argument(
        "--max-evenements",
        type=int,
        default=None,
        help=(
            "Nombre maximum d'événements à indexer "
            f"(défaut : {config.MAX_EVENEMENTS})"
        ),
    )
    args = parser.parse_args(argv)

    setup_logging()
    logger.info("Début de l'indexation")
    nb_evenements, nb_vecteurs = construire_base(args.max_evenements)
    logger.info(
        "Indexation terminée : %d événements, %d vecteurs",
        nb_evenements,
        nb_vecteurs,
    )


if __name__ == "__main__":
    main()
