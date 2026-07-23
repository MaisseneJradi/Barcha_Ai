"""Schémas Pydantic v2 : modèle métier Invoice, état du graphe LangGraph,
et contrats d'entrée/sortie de l'API FastAPI.

Règle transverse (FR-08) : tout champ manquant ou illisible vaut `None`.
Rien n'est inventé.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .config import EXPENSE_CATEGORIES


# --- Modèle métier -----------------------------------------------------------
class LineItem(BaseModel):
    """Une ligne de facture. Tous les champs sont optionnels (peuvent être null)."""

    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None


class Invoice(BaseModel):
    """Facture extraite. Les champs absents/illisibles restent `None` (FR-08)."""

    invoice_number: Optional[str] = None
    issuer_name: Optional[str] = None
    issuer_tax_id: Optional[str] = None          # matricule fiscal de l'émetteur
    client_name: Optional[str] = None
    issue_date: Optional[str] = None             # ISO 'YYYY-MM-DD' si possible
    line_items: List[LineItem] = Field(default_factory=list)
    subtotal_ht: Optional[float] = None
    vat_amount: Optional[float] = None
    total_ttc: Optional[float] = None
    currency: Optional[str] = None

    def dedup_key(self) -> Dict[str, Any]:
        """Clé d'unicité (FR-12) : numéro + matricule + total TTC + date."""
        return {
            "invoice_number": self.invoice_number,
            "issuer_tax_id": self.issuer_tax_id,
            "total_ttc": self.total_ttc,
            "issue_date": self.issue_date,
        }


class ExpenseCategory(str, Enum):
    materiel = "matériel"
    services = "services"
    restauration = "restauration"
    transport = "transport"
    communication = "communication"
    autre = "autre"


# --- Statuts exposés à l'API -------------------------------------------------
class Status(str, Enum):
    completed = "completed"
    en_attente_utilisateur = "en_attente_utilisateur"
    erreur = "erreur"


class PendingType(str, Enum):
    champ_manquant = "champ_manquant"
    doublon = "doublon"


# --- État du graphe LangGraph ------------------------------------------------
class GraphState(TypedDict, total=False):
    """État partagé circulant entre les nœuds LangGraph.

    `total=False` : chaque nœud n'écrit que les clés qu'il modifie.
    """

    user_id: str
    document_id: str
    filename: Optional[str]
    mime: Optional[str]
    file_b64: Optional[str]          # contenu source encodé base64 (entrée OCR)
    activite: Optional[str]          # activité déclarée de l'utilisateur (FR-11)

    ocr_text: str                    # texte de travail (français)
    ocr_text_original: Optional[str] # texte OCR d'origine avant traduction
    detected_language: Optional[str]

    invoice: Dict[str, Any]          # Invoice sérialisée (dict), mise à jour incrémentale
    missing_fields: List[str]

    analysis: Optional[str]
    expense_category: Optional[str]
    deductible: Optional[bool]
    deductibility_reason: Optional[str]

    duplicate_candidate: Optional[Dict[str, Any]]
    duplicate_decision: Optional[str]   # 'confirme' | 'distinct' | None

    saved: Optional[bool]
    duplicate_skipped: Optional[bool]

    status: str                          # statut interne
    error: Optional[str]
    messages: List[Dict[str, str]]


# --- Contrats API ------------------------------------------------------------
class PendingQuestion(BaseModel):
    """Détail d'une interruption HITL (champ manquant ou doublon)."""

    type: PendingType
    question: str
    field: Optional[str] = None
    existing_invoice: Optional[Dict[str, Any]] = None   # doublon : l'existant
    new_invoice: Optional[Dict[str, Any]] = None        # doublon : le nouveau


class AnalyzeResponse(BaseModel):
    status: Status
    thread_id: str
    document_id: Optional[str] = None
    # Rempli si status == completed
    invoice: Optional[Invoice] = None
    analysis: Optional[str] = None
    expense_category: Optional[str] = None
    deductible: Optional[bool] = None
    deductibility_reason: Optional[str] = None
    saved: Optional[bool] = None
    duplicate_skipped: Optional[bool] = None
    # Rempli si status == en_attente_utilisateur
    pending: Optional[PendingQuestion] = None
    # Rempli si status == erreur
    error: Optional[str] = None


class AnswerRequest(BaseModel):
    thread_id: str
    answer: str


class AnswerResponse(BaseModel):
    status: Status
    thread_id: str
    document_id: Optional[str] = None
    # Reprise d'un flux d'analyse (même forme que AnalyzeResponse)
    analyze: Optional[AnalyzeResponse] = None
    # Réponse à une question de suivi (Q&A) sur une facture déjà traitée
    answer: Optional[str] = None
    error: Optional[str] = None


class InvoiceListItem(BaseModel):
    document_id: str
    invoice: Invoice
    analysis: Optional[str] = None
    expense_category: Optional[str] = None
    deductible: Optional[bool] = None
    deductibility_reason: Optional[str] = None
    created_at: Optional[str] = None
