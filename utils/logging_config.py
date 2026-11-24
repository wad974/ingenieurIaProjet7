"""Configuration du logging applicatif."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from . import config

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_configured = False


def _resolve_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    name = (level or config.LOG_LEVEL or "INFO").upper()
    return getattr(logging, name, logging.INFO)


def setup_logging(
    level: str | int | None = None,
    log_file: str | Path | None = None,
) -> None:
    """Configure le logger racine (console + fichier rotatif), une seule fois."""
    global _configured
    if _configured:
        return

    config.ensure_directories()
    root = logging.getLogger()
    root.setLevel(_resolve_level(level))
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    path = Path(log_file) if log_file else Path(config.LOG_FILE)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path,
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:  # disque en lecture seule, droits insuffisants…
        root.warning(
            "Fichier de log indisponible (%s) : journalisation console uniquement",
            exc,
        )

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger nommé (le setup reste à la charge du point d'entrée)."""
    return logging.getLogger(name)
