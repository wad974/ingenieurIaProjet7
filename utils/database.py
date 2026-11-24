"""Persistance des feedbacks utilisateurs (SQLite + SQLAlchemy)."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import DateTime, Integer, Text, create_engine, func, select
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from . import config

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base déclarative SQLAlchemy du projet."""


class Feedback(Base):
    """Un retour utilisateur sur une réponse du chatbot."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    reponse: Mapped[str] = mapped_column(Text, nullable=False, default="")
    sources: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    note: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    commentaire: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )

    def to_dict(self) -> dict[str, Any]:
        """Retourne le feedback sous une forme sérialisable (JSON, DataFrame…)."""
        try:
            sources = json.loads(self.sources or "[]")
        except json.JSONDecodeError:
            sources = []
        return {
            "id": self.id,
            "question": self.question,
            "reponse": self.reponse,
            "sources": sources,
            "note": self.note,
            "commentaire": self.commentaire,
            "created_at": (
                self.created_at.strftime("%Y-%m-%d %H:%M:%S")
                if self.created_at
                else ""
            ),
        }


_factories: dict[str, sessionmaker[Session]] = {}


def _factory_for(db_path: str | Path | None = None) -> sessionmaker[Session]:
    chemin = Path(db_path) if db_path else config.FEEDBACK_DB_PATH
    chemin = chemin.resolve()
    cle = str(chemin)

    if cle not in _factories:
        chemin.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            f"sqlite:///{chemin}",
            connect_args={"check_same_thread": False},
            future=True,
        )
        Base.metadata.create_all(engine)
        _factories[cle] = sessionmaker(bind=engine, expire_on_commit=False)

    return _factories[cle]


@contextmanager
def session_scope(
    db_path: str | Path | None = None,
) -> Iterator[Session]:
    """Fournit une session SQLAlchemy avec commit/rollback automatique."""
    session = _factory_for(db_path)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ajouter_feedback(
    question: str,
    reponse: str,
    note: int,
    commentaire: str = "",
    sources: list[dict[str, Any]] | None = None,
    db_path: str | Path | None = None,
) -> int:
    """Enregistre un feedback et retourne son identifiant."""
    if note not in (-1, 0, 1):
        raise ValueError("note doit valoir -1, 0 ou 1")

    with session_scope(db_path) as session:
        feedback = Feedback(
            question=question.strip(),
            reponse=reponse.strip(),
            note=note,
            commentaire=commentaire.strip(),
            sources=json.dumps(sources or [], ensure_ascii=False),
        )
        session.add(feedback)
        session.flush()
        identifiant = feedback.id

    logger.info("Feedback #%s enregistré (note=%s)", identifiant, note)
    return identifiant


def lister_feedback(
    note: int | None = None,
    limite: int = 500,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Liste les feedbacks, du plus récent au plus ancien."""
    requete = select(Feedback)
    if note in (-1, 1):
        requete = requete.where(Feedback.note == note)
    requete = (
        requete.order_by(Feedback.created_at.desc(), Feedback.id.desc())
        .limit(limite)
    )

    with session_scope(db_path) as session:
        return [ligne.to_dict() for ligne in session.scalars(requete)]


def statistiques_feedback(
    db_path: str | Path | None = None,
) -> dict[str, float | int]:
    """Retourne les compteurs utiles à la page de visualisation."""
    with session_scope(db_path) as session:
        total = session.scalar(select(func.count()).select_from(Feedback)) or 0
        positifs = (
            session.scalar(
                select(func.count())
                .select_from(Feedback)
                .where(Feedback.note == 1)
            )
            or 0
        )
        negatifs = (
            session.scalar(
                select(func.count())
                .select_from(Feedback)
                .where(Feedback.note == -1)
            )
            or 0
        )

    votes = positifs + negatifs
    return {
        "total": total,
        "positifs": positifs,
        "negatifs": negatifs,
        "satisfaction": round(100 * positifs / votes, 1) if votes else 0.0,
    }
