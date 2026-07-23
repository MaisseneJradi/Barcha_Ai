r"""Outil de configuration MongoDB Atlas — construit, TESTE puis écrit MONGODB_URI.

Usage (PowerShell, mot de passe entre GUILLEMETS SIMPLES pour éviter toute
interprétation par le shell) :

    .\.venv\Scripts\python.exe set_mongo.py <user> '<mot_de_passe>' [host]

Exemple :
    .\.venv\Scripts\python.exe set_mongo.py fouratmarouen0_db_user 'MonMotDePasse123' cluster0.nyaufbb.mongodb.net

- Le mot de passe est URL-encodé automatiquement (les caractères spéciaux
  @ : / ? # % … sont donc gérés sans intervention).
- Si le `host` est omis, on réutilise celui déjà présent dans .env.
- Rien n'est écrit tant que le `ping` d'authentification n'a pas réussi.
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote, urlsplit

from pymongo import MongoClient

ENV = Path(__file__).resolve().parent / ".env"


def current_host() -> str | None:
    if not ENV.exists():
        return None
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("MONGODB_URI="):
            return urlsplit(line.split("=", 1)[1].strip()).hostname
    return None


def write_env(uri: str) -> None:
    """Remplace (ou ajoute) la ligne MONGODB_URI= sans toucher au reste du .env."""
    lines = ENV.read_text(encoding="utf-8").splitlines() if ENV.exists() else []
    out, replaced = [], False
    for line in lines:
        if line.strip().startswith("MONGODB_URI="):
            out.append(f"MONGODB_URI={uri}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"MONGODB_URI={uri}")
    ENV.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2

    user = sys.argv[1]
    password = sys.argv[2]
    host = sys.argv[3] if len(sys.argv) > 3 else current_host()
    if not host:
        print("ERREUR : aucun host fourni et aucun trouvé dans .env.")
        return 2

    # userinfo doit être URL-encodé (RFC 3986) : safe='' encode aussi @ : / etc.
    uri = (
        f"mongodb+srv://{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}/?retryWrites=true&w=majority&appName=Cluster0"
    )
    masked = uri.replace(quote(password, safe=""), "*" * 6)
    print(f"URI construite : {masked}")
    print(f"Test d'authentification sur {host} …")

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=8000)
        client.admin.command("ping")
    except Exception as exc:  # noqa: BLE001 - outil CLI : on affiche la cause telle quelle
        print("\n❌ ÉCHEC :", exc)
        print(
            "\nRien n'a été écrit. Pistes :\n"
            "  • Mot de passe erroné → Atlas > Database Access > Edit > Edit Password.\n"
            "  • Utilisateur absent de CE projet (celui qui possède le cluster).\n"
            "  • IP non autorisée → Atlas > Network Access > Add 0.0.0.0/0."
        )
        return 1

    write_env(uri)
    print("\n✅ OK — authentification réussie. MONGODB_URI mis à jour dans .env.")
    print("   Lance maintenant : uvicorn app.api:app --reload --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
