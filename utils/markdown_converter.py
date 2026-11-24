"""Conversion de fichiers et d'URLs en Markdown pour l'ingestion RAG."""

from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify

from . import config
from .data_loader import fiche_evenement

logger = logging.getLogger(__name__)

EXTENSIONS_SUPPORTEES = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".markdown",
    ".html",
    ".htm",
}

_USER_AGENT = "RAG-Ingestion/1.0"


@dataclass
class ConversionResult:
    """Résultat d'une conversion : Markdown + titre lisible."""

    markdown: str
    title: str


# --- Utilitaires ------------------------------------------------------------
def _decoder(contenu: bytes) -> str:
    for encodage in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return contenu.decode(encodage)
        except UnicodeDecodeError:
            continue
    return contenu.decode("utf-8", errors="replace")


def _nettoyer(markdown: str) -> str:
    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip() + "\n"


def _titre_depuis_html(soup: BeautifulSoup) -> str | None:
    if soup.title and soup.title.string:
        titre = soup.title.string.strip()
        if titre:
            return titre
    h1 = soup.find("h1")
    if h1:
        titre = h1.get_text(" ", strip=True)
        if titre:
            return titre
    return None


def _ajouter_titre(markdown: str, titre: str) -> str:
    if markdown.lstrip().startswith("#"):
        return markdown
    return f"# {titre}\n\n{markdown}"


def _titre_depuis_url(url_parsee) -> str:
    """Extrait un titre lisible de l'URL (dataset OpenDataSoft si présent)."""
    correspondance = re.search(r"/datasets/([^/]+)", url_parsee.path)
    if correspondance:
        return correspondance.group(1).replace("-", " ").replace("_", " ").title()
    return url_parsee.netloc + url_parsee.path


# --- Convertisseurs ---------------------------------------------------------
def _pdf_vers_markdown(contenu: bytes, titre: str) -> ConversionResult:
    try:
        import pymupdf
        import pymupdf4llm

        document = pymupdf.open(stream=contenu, filetype="pdf")
        try:
            markdown = pymupdf4llm.to_markdown(document)
        finally:
            document.close()
        if not markdown.strip():
            raise ValueError("PDF sans texte extractible")
        return ConversionResult(
            _nettoyer(_ajouter_titre(markdown, titre)),
            titre,
        )
    except ImportError:  # repli si pymupdf4llm n'est pas installé
        logger.warning("pymupdf4llm absent, repli sur pypdf")
        return _pdf_vers_markdown_pypdf(contenu, titre)


def _pdf_vers_markdown_pypdf(contenu: bytes, titre: str) -> ConversionResult:
    from pypdf import PdfReader

    lecteur = PdfReader(io.BytesIO(contenu))
    pages: list[str] = []
    for numero, page in enumerate(lecteur.pages, start=1):
        texte = (page.extract_text() or "").strip()
        if texte:
            pages.append(f"## Page {numero}\n\n{texte}")
    if not pages:
        raise ValueError("PDF sans texte extractible")
    markdown = f"# {titre}\n\n" + "\n\n".join(pages)
    return ConversionResult(_nettoyer(markdown), titre)


def _docx_vers_markdown(contenu: bytes, titre: str) -> ConversionResult:
    import mammoth

    resultat = mammoth.convert_to_markdown(io.BytesIO(contenu))
    markdown = (resultat.value or "").strip()
    if not markdown:
        raise ValueError("DOCX sans contenu textuel")
    for message in resultat.messages:
        logger.debug("mammoth : %s", message)
    return ConversionResult(
        _nettoyer(_ajouter_titre(markdown, titre)),
        titre,
    )


def _element_vers_markdown(element: Any, numero: int) -> str:
    """Formate un enregistrement JSON en section Markdown."""
    if not isinstance(element, dict):
        return (
            f"## Élément {numero}\n\n"
            f"```json\n{json.dumps(element, ensure_ascii=False, indent=2)}\n```"
        )

    # Enregistrements d'événements OpenAgenda : fiche métier existante.
    if any(
        cle in element
        for cle in ("title_fr", "daterange_fr", "location_city")
    ):
        titre = element.get("title_fr") or f"Événement {numero}"
        return f"## {titre}\n\n{fiche_evenement(element)}"

    titre = (
        element.get("title_fr")
        or element.get("title")
        or element.get("name")
        or element.get("id")
        or f"Élément {numero}"
    )
    lignes = [f"## {titre}"]
    for cle, valeur in element.items():
        if valeur in (None, "", [], {}):
            continue
        if isinstance(valeur, (dict, list)):
            valeur = json.dumps(valeur, ensure_ascii=False)
        lignes.append(f"- **{cle}** : {valeur}")
    return "\n".join(lignes)


def _json_vers_markdown(payload: Any, titre: str) -> ConversionResult:
    """Convertit une réponse JSON en Markdown structuré."""
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        sections = [
            _element_vers_markdown(element, numero)
            for numero, element in enumerate(payload["results"], start=1)
        ]
        markdown = f"# {titre}\n\n" + "\n\n---\n\n".join(sections)
        return ConversionResult(_nettoyer(markdown), titre)

    markdown = (
        f"# {titre}\n\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
    )
    return ConversionResult(_nettoyer(markdown), titre)


def _html_vers_markdown(html: str, titre_par_defaut: str) -> ConversionResult:
    soup = BeautifulSoup(html, "html.parser")
    titre = _titre_depuis_html(soup) or titre_par_defaut

    # Retire le <head> (le <title> est déjà capturé) et les éléments non texte.
    for balise in soup(
        ["script", "style", "noscript", "iframe", "svg", "head"]
    ):
        balise.decompose()

    markdown = markdownify(str(soup), heading_style="ATX")
    markdown = _nettoyer(_ajouter_titre(markdown, titre))
    if not markdown.strip("# \n"):
        raise ValueError("Page HTML sans contenu textuel")
    return ConversionResult(markdown, titre)


def depuis_fichier(nom: str, contenu: bytes) -> ConversionResult:
    """Convertit un fichier uploadé en Markdown selon son extension."""
    suffixe = Path(nom).suffix.lower()
    titre = Path(nom).stem or "Document"

    if not contenu:
        raise ValueError("Fichier vide")

    if suffixe == ".pdf":
        return _pdf_vers_markdown(contenu, titre)
    if suffixe == ".docx":
        return _docx_vers_markdown(contenu, titre)
    if suffixe in {".html", ".htm"}:
        return _html_vers_markdown(_decoder(contenu), titre)
    if suffixe in {".md", ".markdown"}:
        return ConversionResult(_nettoyer(_decoder(contenu)), titre)
    if suffixe in {".txt", ""}:
        return ConversionResult(
            _nettoyer(f"# {titre}\n\n{_decoder(contenu)}"),
            titre,
        )

    raise ValueError(
        f"Extension non supportée : {suffixe or '(aucune)'}. "
        f"Formats acceptés : {', '.join(sorted(EXTENSIONS_SUPPORTEES))}"
    )


def depuis_url(url: str) -> ConversionResult:
    """Télécharge une URL et la convertit en Markdown (HTML, PDF ou texte)."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("L'URL doit commencer par http:// ou https://")

    reponse = requests.get(
        url,
        timeout=config.REQUEST_TIMEOUT,
        headers={"User-Agent": _USER_AGENT},
    )
    reponse.raise_for_status()

    if len(reponse.content) > config.MAX_UPLOAD_BYTES:
        raise ValueError(
            "Réponse trop volumineuse (max "
            f"{config.MAX_UPLOAD_BYTES // (1024 * 1024)} Mo)."
        )

    content_type = reponse.headers.get("content-type", "").lower()
    titre_par_defaut = _titre_depuis_url(parsed)

    if "json" in content_type:
        try:
            payload = reponse.json()
        except ValueError as exc:
            raise ValueError("La réponse JSON est invalide") from exc
        return _json_vers_markdown(payload, titre_par_defaut)

    if "application/pdf" in content_type or parsed.path.lower().endswith(".pdf"):
        return _pdf_vers_markdown(reponse.content, titre_par_defaut)
    if "text/html" in content_type or "application/xhtml" in content_type:
        return _html_vers_markdown(reponse.text, titre_par_defaut)
    if (
        content_type.startswith("text/")
        or "markdown" in content_type
        or not content_type
    ):
        markdown = f"# {titre_par_defaut}\n\n{reponse.text}"
        return ConversionResult(_nettoyer(markdown), titre_par_defaut)

    raise ValueError(f"Type de contenu non supporté : {content_type}")
