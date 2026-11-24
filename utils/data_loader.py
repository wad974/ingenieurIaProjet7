"""Chargement et mise en forme des documents sources (API OpenAgenda)."""

from __future__ import annotations

import logging
from typing import Any

import requests

from . import config

logger = logging.getLogger(__name__)

CHAMPS = (
    "uid,title_fr,description_fr,longdescription_fr,conditions_fr,"
    "daterange_fr,firstdate_begin,lastdate_end,location_name,location_city,"
    "location_address,location_postalcode,canonicalurl,category"
)


def recuperer_evenements(
    max_evenements: int | None = None,
    date_min: str | None = None,
    url: str | None = None,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Récupère les événements publics de La Réunion via l'API OpenDataSoft."""
    max_evenements = max_evenements or config.MAX_EVENEMENTS
    date_min = date_min or config.DATE_MIN
    url = url or config.OPENAGENDA_URL

    http = session or requests.Session()
    resultats: list[dict[str, Any]] = []
    offset = 0
    try:
        while len(resultats) < max_evenements:
            params = {
                "where": (
                    "location_region = 'La Réunion' "
                    f"AND firstdate_begin >= '{date_min}'"
                ),
                "order_by": "firstdate_begin ASC",
                "limit": min(config.PAGE_SIZE, max_evenements - len(resultats)),
                "offset": offset,
                "select": CHAMPS,
            }
            logger.debug(
                "Requête OpenAgenda offset=%s limit=%s",
                offset,
                params["limit"],
            )
            response = http.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
            response.raise_for_status()
            page = response.json().get("results", [])
            if not page:
                break
            resultats.extend(page)
            offset += len(page)
    finally:
        if session is None:
            http.close()

    resultats = resultats[:max_evenements]
    logger.info("%d événements récupérés depuis l'API", len(resultats))
    return resultats


def fiche_evenement(evenement: dict[str, Any]) -> str:
    """Transforme un événement brut en fiche texte indexable."""
    description = (
        evenement.get("description_fr")
        or evenement.get("longdescription_fr")
        or ""
    )
    return "\n".join(
        [
            f"Titre : {evenement.get('title_fr') or 'Sans titre'}",
            f"Date : {evenement.get('daterange_fr') or evenement.get('firstdate_begin') or 'Non précisée'}",
            f"Ville : {evenement.get('location_city') or 'Non précisée'}",
            f"Lieu : {evenement.get('location_name') or 'Non précisé'}",
            f"Catégorie : {evenement.get('category') or 'Non précisée'}",
            f"Conditions : {evenement.get('conditions_fr') or 'Non précisées'}",
            f"Description : {description[:1200]}",
        ]
    )


def charger_evenements(
    max_evenements: int | None = None,
    date_min: str | None = None,
) -> list[dict[str, Any]]:
    """Récupère, filtre et dédoublonne les événements prêts à être indexés."""
    evenements = [
        evenement
        for evenement in recuperer_evenements(max_evenements, date_min)
        if evenement.get("title_fr")
    ]

    uniques: list[dict[str, Any]] = []
    vus: set[tuple[Any, Any]] = set()
    for evenement in evenements:
        cle = (evenement.get("uid"), evenement.get("firstdate_begin"))
        if cle in vus:
            continue
        vus.add(cle)
        uniques.append(evenement)

    logger.info(
        "%d événements uniques après filtrage et dédoublonnage",
        len(uniques),
    )
    return uniques
