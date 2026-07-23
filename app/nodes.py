"""Nœuds LangGraph (un par étape) + routage conditionnel + point d'entrée Q&A.

Les dépendances (client Mistral, base) sont injectées via un conteneur `Deps`
lié aux nœuds par `functools.partial` dans graph.py — ce qui rend les nœuds
testables sans singletons globaux.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from langgraph.types import interrupt  # HITL
from pydantic import ValidationError

from . import prompts
from .config import MANDATORY_FIELDS, MODEL_LARGE, MODEL_SMALL, EXPENSE_CATEGORIES
from .db import Database, DuplicateInvoiceError
from .mistral_client import MistralClient
from .schemas import Invoice, LineItem


@dataclass
class Deps:
    mistral: MistralClient
    db: Database


# ---------------------------------------------------------------------------
# Helpers de coercition (réponses HITL = autorité de l'utilisateur, FR-08)
# ---------------------------------------------------------------------------
_SKIP_WORDS = {"passer", "skip", "ignorer", "aucun", ""}


def _parse_amount(value: str) -> Optional[float]:
    cleaned = re.sub(r"[^\d,.\-]", "", value).replace(" ", "")
    if not cleaned:
        return None
    # virgule décimale française -> point
    if "," in cleaned and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_date(value: str) -> Optional[str]:
    value = value.strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value or None


def _coerce_field(field: str, value: str) -> Any:
    value = value.strip()
    if field in {"total_ttc", "subtotal_ht", "vat_amount"}:
        return _parse_amount(value)
    if field == "issue_date":
        return _parse_date(value)
    return value or None


def _safe_invoice(data: Any) -> Invoice:
    """Construit un Invoice tolérant : sur erreur, on isole ligne par ligne."""
    if not isinstance(data, dict):
        return Invoice()
    try:
        return Invoice.model_validate(data)
    except ValidationError:
        clean = {f: data.get(f) for f in Invoice.model_fields if f != "line_items"}
        inv = Invoice.model_validate(clean)
        items: List[LineItem] = []
        for it in (data.get("line_items") or []):
            try:
                items.append(LineItem.model_validate(it))
            except ValidationError:
                continue
        inv.line_items = items
        return inv


def _compute_missing(inv: Invoice) -> List[str]:
    return [f for f in MANDATORY_FIELDS if getattr(inv, f) in (None, "")]


# ---------------------------------------------------------------------------
# Nœuds du graphe principal
# ---------------------------------------------------------------------------
def ocr_node(state: Dict[str, Any], deps: Deps) -> Dict[str, Any]:
    """OCR de la facture. Le contenu source est passé en base64 dans l'état."""
    import base64

    b64 = state["file_b64"]
    mime = state.get("mime") or "application/pdf"
    data = base64.b64decode(b64)
    text = deps.mistral.ocr(data, mime)
    return {"ocr_text": text, "ocr_text_original": text, "status": "en_cours"}


def detect_language_node(state: Dict[str, Any], deps: Deps) -> Dict[str, Any]:
    system, user = prompts.detect_language(state["ocr_text"])
    # Bascule sur le grand modèle si `small` est saturé (429 capacity).
    result = deps.mistral.chat_json(MODEL_SMALL, system, user, fallback_model=MODEL_LARGE)
    lang = str(result.get("language", "")).lower()[:2] or "fr"
    return {"detected_language": lang}


def translate_to_fr_node(state: Dict[str, Any], deps: Deps) -> Dict[str, Any]:
    """Traduit le texte OCR en français avant tout traitement aval (FR-14)."""
    system, user = prompts.translate_to_fr(state["ocr_text"])
    fr = deps.mistral.chat_text(MODEL_LARGE, system, user)
    return {"ocr_text": fr}  # ocr_text_original conserve la version d'origine


def extract_fields_node(state: Dict[str, Any], deps: Deps) -> Dict[str, Any]:
    system, user = prompts.extract_fields(state["ocr_text"])
    data = deps.mistral.chat_json(MODEL_LARGE, system, user)
    inv = _safe_invoice(data)
    missing = _compute_missing(inv)
    out: Dict[str, Any] = {"invoice": inv.model_dump(), "missing_fields": missing}
    # Suggestions calculées ICI (une seule fois) : le nœud d'interruption se
    # ré-exécute à chaque reprise et ne doit donc PAS relancer d'appel LLM.
    if missing:
        out["field_suggestions"] = _suggest_fields(deps, missing, state["ocr_text"], inv.model_dump())
    return out


def _suggest_fields(deps: Deps, fields: List[str], ocr_text: str, invoice: Dict[str, Any]) -> Dict[str, List[str]]:
    """Propose des valeurs candidates par champ manquant (jamais retenues sans
    approbation humaine). Tolérant aux pannes : en cas d'échec, aucune suggestion
    — le HITL en saisie libre reste disponible."""
    try:
        system, user = prompts.suggest_field_values(fields, ocr_text, invoice)
        result = deps.mistral.chat_json(MODEL_LARGE, system, user)
    except Exception:  # noqa: BLE001 - fonctionnalité d'assistance, non bloquante
        return {}
    out: Dict[str, List[str]] = {}
    for f in fields:
        raw = result.get(f)
        if isinstance(raw, list):
            vals = [str(v).strip() for v in raw if str(v).strip()]
        elif isinstance(raw, (str, int, float)) and str(raw).strip():
            vals = [str(raw).strip()]
        else:
            vals = []
        if vals:
            out[f] = vals[:3]
    return out


def ask_missing_field_node(state: Dict[str, Any], deps: Deps) -> Dict[str, Any]:
    """Interrompt le graphe pour demander UN champ manquant à l'utilisateur.

    >>> ZONE DÉLICATE #2 — interruption LangGraph (human-in-the-loop) <<<
    `interrupt(payload)` suspend l'exécution et remonte `payload` à l'appelant
    (l'API renvoie alors un état `en_attente_utilisateur`). Le thread est figé
    par le checkpointer. Sur `POST /answer`, on reprend avec
    `Command(resume=<réponse>)` : LangGraph RÉEXÉCUTE ce nœud depuis le début,
    et cette fois `interrupt(...)` RENVOIE la valeur de reprise au lieu de
    suspendre. Tout code situé avant `interrupt` doit donc rester sans effet de
    bord (ici : simple lecture d'état). Un champ est traité par tour ; le
    routage reboucle tant qu'il reste des champs manquants.
    """
    missing = list(state.get("missing_fields") or [])
    invoice = dict(state.get("invoice") or {})
    if not missing:
        return {}

    field = missing[0]
    question = prompts.ask_missing_field(field, invoice)
    # Lecture PURE de l'état (suggestions pré-calculées) : sûr à la ré-exécution.
    suggestions = (state.get("field_suggestions") or {}).get(field, [])
    answer = interrupt(
        {"type": "champ_manquant", "field": field, "question": question, "suggestions": suggestions}
    )

    remaining = missing[1:]
    if answer is not None and str(answer).strip().lower() not in _SKIP_WORDS:
        invoice[field] = _coerce_field(field, str(answer))
    return {"invoice": invoice, "missing_fields": remaining}


def write_analysis_node(state: Dict[str, Any], deps: Deps) -> Dict[str, Any]:
    system, user = prompts.write_analysis(state["ocr_text"], state["invoice"])
    analysis = deps.mistral.chat_text(MODEL_LARGE, system, user)
    return {"analysis": analysis}


def classify_expense_node(state: Dict[str, Any], deps: Deps) -> Dict[str, Any]:
    system, user = prompts.classify_expense(state["ocr_text"], state["invoice"])
    # Bascule sur le grand modèle si `small` est saturé (429 capacity).
    result = deps.mistral.chat_json(MODEL_SMALL, system, user, fallback_model=MODEL_LARGE)
    category = str(result.get("category", "")).strip().lower()
    if category not in EXPENSE_CATEGORIES:
        category = "autre"
    return {"expense_category": category}


def assess_deductibility_node(state: Dict[str, Any], deps: Deps) -> Dict[str, Any]:
    # 'activite' (activité déclarée de l'utilisateur) est fournie dans l'état
    # initial par l'orchestrateur / le dashboard ; optionnelle.
    activite = state.get("activite", "")
    system, user = prompts.assess_deductibility(state["ocr_text"], state["invoice"], activite)
    result = deps.mistral.chat_json(MODEL_LARGE, system, user)
    deductible = bool(result.get("deductible", False))
    reason = str(result.get("reason", "")).strip() or None
    return {"deductible": deductible, "deductibility_reason": reason}


def check_duplicate_node(state: Dict[str, Any], deps: Deps) -> Dict[str, Any]:
    """Recherche un doublon (FR-12). Si trouvé : interruption pour confirmation
    humaine — jamais de rejet automatique."""
    inv = _safe_invoice(state.get("invoice") or {})
    existing = deps.db.find_duplicate(state["user_id"], inv.dedup_key())
    if not existing:
        return {"duplicate_candidate": None, "duplicate_decision": "distinct"}

    existing_clean = {k: v for k, v in existing.items() if k != "_id"}
    decision = interrupt(
        {
            "type": "doublon",
            "question": (
                "Une facture très similaire existe déjà. S'agit-il d'un doublon ? "
                "(répondez « oui » pour ignorer, « non » pour l'enregistrer quand même)"
            ),
            "existing_invoice": existing_clean.get("invoice", existing_clean),
            "new_invoice": state.get("invoice"),
        }
    )
    d = str(decision).strip().lower()
    confirme = d in {"oui", "o", "yes", "y", "confirmer", "doublon", "true", "1"}
    return {
        "duplicate_candidate": existing_clean,
        "duplicate_decision": "confirme" if confirme else "distinct",
    }


def save_to_db_node(state: Dict[str, Any], deps: Deps) -> Dict[str, Any]:
    """Persiste la facture (sauf doublon confirmé) et initialise la session chat."""
    if state.get("duplicate_decision") == "confirme":
        return {"status": "completed", "saved": False, "duplicate_skipped": True}

    inv = _safe_invoice(state.get("invoice") or {})
    doc = {
        "user_id": state["user_id"],
        "document_id": state["document_id"],
        "invoice": inv.model_dump(),
        "analysis": state.get("analysis"),
        "expense_category": state.get("expense_category"),
        "deductible": state.get("deductible"),
        "deductibility_reason": state.get("deductibility_reason"),
        "ocr_text": state.get("ocr_text"),
        "ocr_text_original": state.get("ocr_text_original"),
        "detected_language": state.get("detected_language"),
        # Champs de la clé unique remontés au niveau racine (index UNIQUE).
        "invoice_number": inv.invoice_number,
        "issuer_tax_id": inv.issuer_tax_id,
        "total_ttc": inv.total_ttc,
        "issue_date": inv.issue_date,
    }
    try:
        deps.db.insert_invoice(doc)
    except DuplicateInvoiceError:
        # Course entre le check et l'insert : on traite comme doublon.
        return {"status": "completed", "saved": False, "duplicate_skipped": True}

    if state.get("analysis"):
        deps.db.append_messages(
            state["user_id"],
            state["document_id"],
            [{"role": "assistant", "content": state["analysis"]}],
        )
    return {"status": "completed", "saved": True, "duplicate_skipped": False}


# ---------------------------------------------------------------------------
# Fonctions de routage (arêtes conditionnelles)
# ---------------------------------------------------------------------------
def route_after_detect(state: Dict[str, Any]) -> str:
    lang = (state.get("detected_language") or "fr").lower()
    return "extract_fields" if lang.startswith("fr") else "translate_to_fr"


def route_after_extract(state: Dict[str, Any]) -> str:
    return "ask_missing_field" if state.get("missing_fields") else "write_analysis"


def route_after_ask(state: Dict[str, Any]) -> str:
    return "ask_missing_field" if state.get("missing_fields") else "write_analysis"


# ---------------------------------------------------------------------------
# Point d'entrée Q&A séparé (ancré sur OCR stocké + historique, SANS RAG)
# ---------------------------------------------------------------------------
def answer_question(deps: Deps, user_id: str, document_id: str, question: str) -> str:
    doc = deps.db.get_invoice_by_document_id(user_id, document_id)
    if not doc:
        raise ValueError("Facture introuvable pour cette session.")
    history = deps.db.get_history(user_id, document_id)
    system, user = prompts.qa_answer(doc.get("ocr_text", ""), doc.get("invoice", {}), history, question)
    answer = deps.mistral.chat_text(MODEL_LARGE, system, user)
    deps.db.append_messages(
        user_id,
        document_id,
        [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
    )
    return answer
