"""Tous les prompts LLM, centralisés et rédigés en français.

Chaque fonction renvoie un couple (system, user). La sortie utilisateur finale
est systématiquement en français (FR-14).
"""
from __future__ import annotations

from typing import Dict, List

from .config import EXPENSE_CATEGORIES

_CATS = ", ".join(EXPENSE_CATEGORIES)


# --- Détection de langue -----------------------------------------------------
def detect_language(ocr_text: str):
    system = (
        "Tu es un détecteur de langue. Tu réponds STRICTEMENT en JSON "
        '{"language": "<code ISO 639-1>"} et rien d\'autre.'
    )
    user = f"Quelle est la langue principale de ce texte de facture ?\n\n{ocr_text[:4000]}"
    return system, user


# --- Traduction vers le français --------------------------------------------
def translate_to_fr(ocr_text: str):
    system = (
        "Tu es un traducteur professionnel. Traduis fidèlement en français le "
        "texte de facture fourni, en conservant les montants, dates, numéros et "
        "la mise en page (lignes, tableaux) à l'identique. Ne commente pas, "
        "ne résume pas : renvoie uniquement la traduction."
    )
    user = ocr_text
    return system, user


# --- Extraction structurée ---------------------------------------------------
def extract_fields(ocr_text: str):
    system = (
        "Tu es un extracteur d'informations de factures. À partir du texte OCR, "
        "tu renvoies STRICTEMENT un objet JSON respectant ce schéma :\n"
        "{\n"
        '  "invoice_number": string|null,\n'
        '  "issuer_name": string|null,\n'
        '  "issuer_tax_id": string|null,\n'
        '  "client_name": string|null,\n'
        '  "issue_date": string|null,   // format ISO YYYY-MM-DD si possible\n'
        '  "line_items": [ {"description": string|null, "quantity": number|null, '
        '"unit_price": number|null, "total": number|null} ],\n'
        '  "subtotal_ht": number|null,\n'
        '  "vat_amount": number|null,\n'
        '  "total_ttc": number|null,\n'
        '  "currency": string|null\n'
        "}\n"
        "RÈGLES IMPÉRATIVES :\n"
        "- Comprends le SENS de chaque valeur d'après le contexte et la mise en "
        "page, même sans libellé explicite (ex. reconnais 12/02/2026 comme la "
        "date d'émission sans que le mot « date » apparaisse).\n"
        "- Si une valeur est absente, illisible ou réellement ambiguë : mets "
        "null. N'INVENTE JAMAIS, ne devine pas.\n"
        "- Les montants sont des nombres (pas de symbole monétaire)."
    )
    user = f"Texte OCR de la facture :\n\n{ocr_text}"
    return system, user


# --- Question de champ manquant (HITL) --------------------------------------
def ask_missing_field(field: str, invoice: Dict) -> str:
    """Formule, en français, la question posée à l'utilisateur pour un champ."""
    libelles = {
        "invoice_number": "le numéro de la facture",
        "issuer_name": "le nom de l'émetteur (fournisseur)",
        "issuer_tax_id": "le matricule fiscal / SIREN de l'émetteur",
        "issue_date": "la date d'émission (JJ/MM/AAAA)",
        "total_ttc": "le montant total TTC",
        "subtotal_ht": "le sous-total HT",
        "vat_amount": "le montant de TVA",
        "currency": "la devise",
        "client_name": "le nom du client",
    }
    libelle = libelles.get(field, field)
    return (
        f"Je n'ai pas pu lire {libelle} sur la facture. "
        f"Pouvez-vous me le préciser ? (répondez « passer » pour l'ignorer)"
    )


# --- Analyse rédigée (une analyse, pas un résumé) ----------------------------
def write_analysis(ocr_text: str, invoice: Dict):
    system = (
        "Tu es un assistant comptable pour micro-entrepreneurs français "
        "(créateurs de contenu). Rédige une COURTE ANALYSE en français de cette "
        "facture — PAS un résumé. Ne répète pas la liste des champs : explique "
        "ce que cette dépense signifie pour l'activité, ce qui est notable "
        "(montant, nature, TVA, régularité), et ce à quoi il faut faire "
        "attention. 3 à 5 phrases, ton professionnel et concret."
    )
    user = (
        f"Champs extraits : {invoice}\n\n"
        f"Texte de la facture :\n{ocr_text[:6000]}"
    )
    return system, user


# --- Classification de la nature de dépense ---------------------------------
def classify_expense(ocr_text: str, invoice: Dict):
    system = (
        "Tu classes la nature d'une dépense de facture dans EXACTEMENT une "
        f"catégorie parmi : {_CATS}.\n"
        'Réponds STRICTEMENT en JSON : {"category": "<une des catégories>"}.'
    )
    user = f"Champs : {invoice}\n\nExtrait :\n{ocr_text[:3000]}"
    return system, user


# --- Évaluation de déductibilité --------------------------------------------
def assess_deductibility(ocr_text: str, invoice: Dict, activite: str):
    system = (
        "Tu évalues si une dépense est DÉDUCTIBLE pour un micro-entrepreneur "
        "français, au sens : elle doit servir l'activité de l'entreprise et non "
        "un intérêt personnel.\n"
        "NUANCE MICRO À RAPPELER dans la justification : sous le régime micro, "
        "l'abattement forfaitaire remplace la déduction réelle — une dépense "
        "« déductible » NE réduit donc PAS la base imposable. Le drapeau reste "
        "utile pour séparer pro/perso et pour les utilisateurs au régime réel.\n"
        'Réponds STRICTEMENT en JSON : {"deductible": true|false, '
        '"reason": "<une phrase en français>"}.'
    )
    user = (
        f"Activité déclarée de l'utilisateur : {activite or 'non précisée'}\n"
        f"Champs : {invoice}\n\nExtrait :\n{ocr_text[:3000]}"
    )
    return system, user


# --- Q&A sur une facture déjà traitée (ancrage OCR + historique, sans RAG) ---
def qa_answer(ocr_text: str, invoice: Dict, history: List[Dict[str, str]], question: str):
    system = (
        "Tu réponds en français aux questions d'un utilisateur sur UNE facture "
        "déjà analysée. Tu t'appuies UNIQUEMENT sur le texte OCR de cette "
        "facture et sur l'historique de conversation fournis. Si l'information "
        "n'y figure pas, dis-le clairement : n'invente rien, ne va chercher "
        "aucune source externe."
    )
    hist = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in history) or "(aucun)"
    user = (
        f"Champs extraits : {invoice}\n\n"
        f"Texte OCR de la facture :\n{ocr_text}\n\n"
        f"Historique de conversation :\n{hist}\n\n"
        f"Question de l'utilisateur : {question}"
    )
    return system, user
