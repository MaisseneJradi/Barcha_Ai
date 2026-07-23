"""Doublures de test : client Mistral factice + fabriques de graphe/base.

Aucune clé API ni MongoDB réel n'est requis : le LLM est simulé de façon
déterministe (branchement sur des marqueurs de prompt) et la base utilise
`mongomock`. Le checkpointer est un `MemorySaver` en mémoire de processus.
"""
from __future__ import annotations

import base64
import uuid
from typing import Any, Dict, List, Optional, Tuple

import mongomock
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt

from app.db import Database
from app.graph import build_graph
from app.nodes import Deps


class FakeMistral:
    """Simule le client Mistral en se branchant sur le contenu des prompts."""

    def __init__(
        self,
        *,
        invoice: Dict[str, Any],
        language: str = "fr",
        category: str = "services",
        deductible: bool = True,
        reason: str = "Dépense liée à l'activité de création de contenu.",
        analysis: str = "Cette dépense correspond à un service professionnel récurrent.",
        translation: Optional[str] = None,
        qa: str = "D'après la facture, le montant TTC est bien celui indiqué.",
        ocr_text: str = "FACTURE — texte OCR de test\nTotal TTC ...",
        suggestions: Optional[Dict[str, List[str]]] = None,
    ):
        self.invoice = invoice
        self.language = language
        self.category = category
        self.deductible = deductible
        self.reason = reason
        self.analysis = analysis
        self.translation = translation
        self.qa = qa
        self.ocr_text = ocr_text
        # Candidats proposés pour les champs manquants (HITL assisté).
        self.suggestions = suggestions or {}

    def ocr(self, data: bytes, mime: str) -> str:
        return self.ocr_text

    def chat_json(
        self, model: str, system: str, user: str, *,
        temperature: float = 0.0, fallback_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        if "détecteur de langue" in system:
            return {"language": self.language}
        if "CANDIDATES" in system:  # prompt de suggestions de champs manquants
            return dict(self.suggestions)
        if "classes la nature" in system:
            return {"category": self.category}
        if "extracteur d'informations" in system:
            return dict(self.invoice)
        if "DÉDUCTIBLE" in system:
            return {"deductible": self.deductible, "reason": self.reason}
        raise AssertionError(f"Prompt chat_json inattendu : {system[:60]!r}")

    def chat_text(
        self, model: str, system: str, user: str, *,
        temperature: float = 0.0, fallback_model: Optional[str] = None,
    ) -> str:
        if "traducteur" in system:
            return self.translation if self.translation is not None else user
        if "COURTE ANALYSE" in system:
            return self.analysis
        if "questions d'un utilisateur sur UNE facture" in system:
            return self.qa
        raise AssertionError(f"Prompt chat_text inattendu : {system[:60]!r}")


def make_db() -> Database:
    db = Database(mongomock.MongoClient(), "testdb")
    db.ensure_indexes()
    return db


def make_graph(mistral: FakeMistral, db: Database) -> Tuple[Any, Deps]:
    deps = Deps(mistral=mistral, db=db)
    return build_graph(deps, MemorySaver()), deps


def initial_state(user_id: str, *, activite: str = "influenceur BNC") -> Dict[str, Any]:
    return {
        "user_id": user_id,
        "document_id": uuid.uuid4().hex,
        "filename": "facture.pdf",
        "mime": "application/pdf",
        "file_b64": base64.b64encode(b"dummy-bytes").decode("ascii"),
        "activite": activite,
        "messages": [],
    }


def run(graph, payload, thread_id: str) -> Dict[str, Any]:
    """Invoque le graphe ; l'interruption HITL n'est pas une erreur."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        graph.invoke(payload, config)
    except GraphInterrupt:
        pass
    return config


def inspect(graph, config) -> Tuple[Dict[str, Any], List[Any]]:
    """Retourne (valeurs d'état, liste d'interruptions en attente)."""
    snap = graph.get_state(config)
    ints = [i for t in snap.tasks for i in (getattr(t, "interrupts", None) or [])]
    return (snap.values or {}), ints


COMPLETE_INVOICE: Dict[str, Any] = {
    "invoice_number": "F-2026-001",
    "issuer_name": "ACME Studio",
    "issuer_tax_id": "FR12345678901",
    "client_name": "Créateur Solo",
    "issue_date": "2026-02-12",
    "line_items": [{"description": "Abonnement outil de montage", "quantity": 1, "unit_price": 100.0, "total": 100.0}],
    "subtotal_ht": 100.0,
    "vat_amount": 20.0,
    "total_ttc": 120.0,
    "currency": "EUR",
}
