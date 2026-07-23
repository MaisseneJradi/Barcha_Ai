"""Configuration centrale de l'Agent 3.

Charge la configuration depuis les variables d'environnement (fichier `.env`).
Seules deux variables sont requises pour tourner de bout en bout :
`MISTRAL_API_KEY` et `MONGODB_URI`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

# Charge .env une seule fois à l'import du module.
load_dotenv()

# --- Modèles Mistral imposés par le cahier des charges -----------------------
MODEL_OCR = "mistral-ocr-latest"          # OCR des factures
MODEL_LARGE = "mistral-large-latest"      # extraction / analyse / Q&A / traduction
MODEL_SMALL = "mistral-small-latest"      # classification (moins coûteux)

# Catégories de dépense autorisées (FR-10). Toute autre valeur est invalide.
EXPENSE_CATEGORIES: List[str] = [
    "matériel",
    "services",
    "restauration",
    "transport",
    "communication",
    "autre",
]

# Champs obligatoires pour lever la boucle human-in-the-loop (FR-08 / FR-12).
# Ce sont l'identité de la facture + la clé de déduplication (hors tax_id, qui
# est souvent absent de factures pourtant valides et reste donc « skippable »).
MANDATORY_FIELDS: List[str] = [
    "invoice_number",
    "issuer_name",
    "issue_date",
    "total_ttc",
]

# Nombre maximum de tours de questions pour éviter une boucle infinie.
MAX_MISSING_FIELD_ROUNDS = 12


@dataclass
class Settings:
    """Paramètres runtime résolus depuis l'environnement."""

    mistral_api_key: str = field(default_factory=lambda: os.getenv("MISTRAL_API_KEY", ""))
    mongodb_uri: str = field(default_factory=lambda: os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    mongodb_db: str = field(default_factory=lambda: os.getenv("MONGODB_DB", "ledgermind"))
    # Nom de la base utilisée par le checkpointer LangGraph (même instance Mongo).
    checkpoint_db: str = field(default_factory=lambda: os.getenv("CHECKPOINT_DB", "ledgermind_checkpoints"))

    def require_api_key(self) -> str:
        if not self.mistral_api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY manquante. Renseignez-la dans le fichier .env "
                "(voir .env.example)."
            )
        return self.mistral_api_key


def get_settings() -> Settings:
    """Retourne une instance fraîche des paramètres (relit l'environnement)."""
    return Settings()
