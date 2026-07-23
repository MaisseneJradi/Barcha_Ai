"""Couche MongoDB : client, index, CRUD factures + mémoire de conversation.

Deux collections (FR-12 / FR-13) :
  - `invoices`      : facture extraite + analyse + classification + déductibilité.
                      Index UNIQUE sur (invoice_number, issuer_tax_id, total_ttc, issue_date).
                      Index sur user_id.
  - `chat_sessions` : historique de conversation par (user_id, document_id).

Le client Mongo est injectable pour permettre les tests avec `mongomock`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import ASCENDING, MongoClient
from pymongo.errors import DuplicateKeyError, OperationFailure

# Nom de l'index unique de déduplication (référencé à l'insertion).
UNIQUE_INDEX_NAME = "uniq_invoice_dedup_key"


class DuplicateInvoiceError(Exception):
    """Levée quand l'insertion viole l'index unique de déduplication."""


class Database:
    def __init__(self, client: MongoClient, db_name: str):
        self._client = client
        self._db = client[db_name]
        self.invoices = self._db["invoices"]
        self.chat_sessions = self._db["chat_sessions"]

    @classmethod
    def connect(cls, uri: str, db_name: str) -> "Database":
        return cls(MongoClient(uri), db_name)

    def ensure_indexes(self) -> None:
        """Crée les index requis (idempotent, auto-réparant)."""
        # Index UNIQUE de déduplication (FR-12).
        self._ensure_index(
            self.invoices,
            [
                ("invoice_number", ASCENDING),
                ("issuer_tax_id", ASCENDING),
                ("total_ttc", ASCENDING),
                ("issue_date", ASCENDING),
            ],
            name=UNIQUE_INDEX_NAME,
            unique=True,
        )
        # Index de listing par utilisateur (FR-13).
        self._ensure_index(self.invoices, [("user_id", ASCENDING)], name="idx_user_id")
        # Historique de chat par (user_id, document_id).
        self._ensure_index(
            self.chat_sessions,
            [("user_id", ASCENDING), ("document_id", ASCENDING)],
            name="uniq_chat_session",
            unique=True,
        )

    @staticmethod
    def _ensure_index(collection, keys, name, unique: bool = False) -> None:
        """Crée un index, en tolérant qu'un index équivalent préexiste.

        Si la même clé existe déjà sous un AUTRE nom (typiquement une base créée
        par une version antérieure), MongoDB lève OperationFailure code 85
        (IndexOptionsConflict). On supprime alors l'ancien index et on recrée
        celui attendu : le démarrage reste idempotent d'une version à l'autre.
        """
        try:
            collection.create_index(keys, name=name, unique=unique)
            return
        except OperationFailure as exc:
            if exc.code != 85:  # 85 = IndexOptionsConflict ; tout autre code = vraie erreur
                raise
        # Trouver l'index préexistant portant exactement la même clé et le remplacer.
        desired = [(field, direction) for field, direction in keys]
        for idx_name, info in collection.index_information().items():
            if idx_name == "_id_":
                continue
            if [(f, d) for f, d in info.get("key", [])] == desired:
                collection.drop_index(idx_name)
                break
        collection.create_index(keys, name=name, unique=unique)

    # -- Déduplication --------------------------------------------------------
    def find_duplicate(self, user_id: str, dedup_key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Cherche une facture existante du même utilisateur avec la même clé."""
        query = {"user_id": user_id, **{k: dedup_key.get(k) for k in (
            "invoice_number", "issuer_tax_id", "total_ttc", "issue_date")}}
        return self.invoices.find_one(query)

    # -- Persistance factures -------------------------------------------------
    def insert_invoice(self, doc: Dict[str, Any]) -> str:
        """Insère une facture ; lève DuplicateInvoiceError si la clé existe déjà."""
        payload = dict(doc)
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        try:
            res = self.invoices.insert_one(payload)
            return str(res.inserted_id)
        except DuplicateKeyError as exc:  # course entre check et insert
            raise DuplicateInvoiceError(str(exc)) from exc

    def list_invoices(self, user_id: str) -> List[Dict[str, Any]]:
        cursor = self.invoices.find({"user_id": user_id}).sort("created_at", ASCENDING)
        out: List[Dict[str, Any]] = []
        for d in cursor:
            d.pop("_id", None)
            out.append(d)
        return out

    def get_invoice_by_document_id(self, user_id: str, document_id: str) -> Optional[Dict[str, Any]]:
        d = self.invoices.find_one({"user_id": user_id, "document_id": document_id})
        if d:
            d.pop("_id", None)
        return d

    # -- Mémoire de conversation ---------------------------------------------
    def get_history(self, user_id: str, document_id: str) -> List[Dict[str, str]]:
        doc = self.chat_sessions.find_one({"user_id": user_id, "document_id": document_id})
        return list(doc.get("messages", [])) if doc else []

    def append_messages(self, user_id: str, document_id: str, messages: List[Dict[str, str]]) -> None:
        """Ajoute des messages à l'historique (crée la session si absente)."""
        self.chat_sessions.update_one(
            {"user_id": user_id, "document_id": document_id},
            {
                "$push": {"messages": {"$each": messages}},
                "$setOnInsert": {"created_at": datetime.now(timezone.utc).isoformat()},
            },
            upsert=True,
        )
