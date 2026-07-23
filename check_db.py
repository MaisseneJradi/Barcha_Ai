r"""Diagnostic MongoDB Atlas de l'Agent 3.

Affiche ce qui est RÉELLEMENT stocké : bases, collections, nombre de factures,
et un aperçu de chaque facture (user_id / document_id / clé de dédup).

    .\.venv\Scripts\python.exe check_db.py
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(override=True)

URI = os.environ["MONGODB_URI"]
DB = os.getenv("MONGODB_DB", "ledgermind")
CKPT = os.getenv("CHECKPOINT_DB", "ledgermind_checkpoints")

client = MongoClient(URI, serverSelectionTimeoutMS=8000)
client.admin.command("ping")
print("✅ Connexion Atlas OK\n")

print("Bases présentes :", client.list_database_names(), "\n")

db = client[DB]
print(f"— Base métier « {DB} » —")
print("  Collections :", db.list_collection_names())
inv = db["invoices"]
n = inv.count_documents({})
print(f"  invoices : {n} document(s)")
for d in inv.find({}, {"user_id": 1, "document_id": 1, "invoice_number": 1,
                       "issuer_tax_id": 1, "total_ttc": 1, "issue_date": 1, "_id": 0}).limit(20):
    print("    •", d)
chat = db["chat_sessions"]
print(f"  chat_sessions : {chat.count_documents({})} session(s)")

print(f"\n— Base checkpoints « {CKPT} » —")
ckpt = client[CKPT]
print("  Collections :", ckpt.list_collection_names())
for c in ckpt.list_collection_names():
    print(f"    {c} : {ckpt[c].count_documents({})} doc(s)")

print("\nSi invoices = 0 mais checkpoints > 0 : le graphe tourne mais la "
      "sauvegarde échoue/est ignorée (doublon).")
