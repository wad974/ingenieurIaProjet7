"""Flux 2 — injection de fichiers et/ou d'URLs dans l'index RAG."""

from __future__ import annotations

import io
import logging
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BeforeValidator

from api.schemas import IngestionOut
from utils import config
from utils.document_ingestion import SourceIngeree, ingerer_sources
from utils.markdown_converter import depuis_fichier, depuis_url

router = APIRouter(tags=["2. Ingestion"])
logger = logging.getLogger(__name__)


def _erreur(source: str, kind: str, message: str) -> dict[str, Any]:
    return {
        "source": source,
        "title": None,
        "kind": kind,
        "status": "error",
        "characters": 0,
        "chunks": 0,
        "markdown_path": None,
        "error": message,
    }


def _champ_fichier_tolerant(value: Any) -> Any:
    """Convertit `files=""` (Swagger sans fichier) en fichier vide."""
    if isinstance(value, str):
        if value.strip():
            raise ValueError(
                "Ce champ attend un fichier uploadé (multipart/form-data), "
                "pas une chaîne de caractères."
            )
        return UploadFile(filename="", file=io.BytesIO(b""))
    return value


# `UploadFile` reste le type exposé : Swagger conserve son sélecteur de
# fichier, et un champ vide est accepté puis ignoré.
FichierTolerant = Annotated[UploadFile, BeforeValidator(_champ_fichier_tolerant)]


@router.post(
    "/ingest",
    response_model=IngestionOut,
    summary="Injecter des fichiers et/ou URLs (Markdown -> chunks -> FAISS)",
    description=(
        "Convertit chaque source en Markdown, la découpe, calcule les "
        "embeddings Ollama et met à jour l'index FAISS. Formats fichiers : "
        "PDF, DOCX, TXT, MD, HTML. Une URL peut pointer vers une page HTML, "
        "un PDF ou un fichier texte."
    ),
)
def ingerer(
    files: list[FichierTolerant] | None = File(
        default=None,
        description="Fichiers à ingérer (PDF, DOCX, TXT, MD, HTML).",
    ),
    urls: list[str] | None = Form(
        default=None,
        description="URLs http(s) à ingérer (répéter le champ pour plusieurs).",
    ),
    replace: bool = Form(
        default=False,
        description="Reconstruire l'index depuis zéro au lieu d'ajouter.",
    ),
) -> IngestionOut:
    liens = [url.strip() for url in (urls or []) if url.strip()]

    # Les champs `files=""` deviennent des UploadFile vides : on les ignore.
    fichiers = [element for element in (files or []) if element.filename]
    echecs: list[dict[str, Any]] = []

    if not fichiers and not liens:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Aucun fichier ni URL fourni. Sélectionne un fichier dans "
                    "`files` ou renseigne `urls`."
                ),
                "sources": echecs,
            },
        )

    sources: list[SourceIngeree] = []

    for upload in fichiers:
        nom = upload.filename or "fichier"
        try:
            contenu = upload.file.read(config.MAX_UPLOAD_BYTES + 1)
            if len(contenu) > config.MAX_UPLOAD_BYTES:
                raise ValueError(
                    "Fichier trop volumineux (max "
                    f"{config.MAX_UPLOAD_BYTES // (1024 * 1024)} Mo)."
                )
            conversion = depuis_fichier(nom, contenu)
            sources.append(
                SourceIngeree(
                    title=conversion.title,
                    source=nom,
                    kind="file",
                    markdown=conversion.markdown,
                )
            )
        except Exception as exc:
            logger.warning("Échec de conversion du fichier %s : %s", nom, exc)
            echecs.append(_erreur(nom, "file", str(exc)))
        finally:
            upload.file.close()

    for url in liens:
        try:
            conversion = depuis_url(url)
            sources.append(
                SourceIngeree(
                    title=conversion.title,
                    source=url,
                    kind="url",
                    markdown=conversion.markdown,
                )
            )
        except Exception as exc:
            logger.warning("Échec de conversion de l'URL %s : %s", url, exc)
            echecs.append(_erreur(url, "url", str(exc)))

    if not sources:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Aucune source n'a pu être convertie.",
                "sources": echecs,
            },
        )

    try:
        resultat = ingerer_sources(sources, replace=replace)
    except Exception as exc:
        logger.exception("Échec de l'ingestion")
        raise HTTPException(
            status_code=500,
            detail=f"Échec de l'ingestion : {exc}",
        ) from exc

    resultat["sources"] = list(resultat["sources"]) + echecs
    return IngestionOut(**resultat)
