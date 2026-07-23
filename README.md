# LedgerMind — Agent 3 : Analyse documentaire (factures)

Service FastAPI autonome qui extrait, analyse, classe, évalue la déductibilité
et déduplique des **factures** (`facture` uniquement), avec boucle
*human-in-the-loop* et mémoire de conversation. Un des agents de la plateforme
multi-agents LedgerMind ; consommé par le dashboard (upload) **et** par
l'orchestrateur central (function calling HTTP).

## Ce que fait l'agent

`ocr → detect_language → (translate_to_fr) → extract_fields → (ask_missing_field, HITL) → write_analysis → classify_expense → assess_deductibility → check_duplicate → (HITL doublon) → save_to_db`

Un point d'entrée **Q&A** séparé répond aux questions de suivi sur une facture
déjà traitée, ancré **uniquement** sur l'OCR stocké + l'historique de chat.

- **LLM** : Mistral (`mistral-ocr-latest`, `mistral-large-latest`, `mistral-small-latest`).
- **Orchestration** : LangGraph (graphe à état + `interrupt` HITL + checkpointer).
- **Base** : MongoDB (`pymongo`).
- **Aucun RAG, aucun vector store, aucun embedding** : l'OCR complet tient dans
  le contexte et est passé directement au LLM.
- **Sortie systématiquement en français** (détection + traduction si besoin).

## Stack & pré-requis

- **Python 3.11+** (exigence du cahier des charges). Le code évite volontairement
  la syntaxe propre à 3.10+ et reste importable sur 3.9, mais visez 3.11+.
- **MongoDB local** en écoute sur `mongodb://localhost:27017` (ou Docker :
  `docker run -d -p 27017:27017 mongo:7`). *Non requis pour les tests.*
- Une **clé API Mistral** (`console.mistral.ai`). *Non requise pour les tests.*

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # puis renseigner MISTRAL_API_KEY
```

## Variables d'environnement (`.env`)

| Variable          | Requis | Défaut                          |
|-------------------|--------|---------------------------------|
| `MISTRAL_API_KEY` | oui    | —                               |
| `MONGODB_URI`     | non    | `mongodb://localhost:27017`     |
| `MONGODB_DB`      | non    | `ledgermind`                    |
| `CHECKPOINT_DB`   | non    | `ledgermind_checkpoints`        |

## Lancer le service

```bash
uvicorn app.api:app --reload --port 8000
# Documentation interactive : http://localhost:8000/docs
```

## Endpoints

| Méthode | Route                 | Corps                                  | Rôle |
|---------|-----------------------|----------------------------------------|------|
| GET     | `/health`             | —                                      | Vérifie que le service tourne. |
| POST    | `/analyze`            | `multipart`: `user_id`, `file`, `activite?` | Lance le graphe complet. |
| POST    | `/answer`             | `{ "thread_id": "...", "answer": "..." }` | Répond à un champ manquant, **confirme un doublon**, ou pose une question de suivi. |
| GET     | `/invoices/{user_id}` | —                                      | Liste les factures d'un utilisateur. |

Chaque réponse porte un `status` ∈ `completed` / `en_attente_utilisateur` / `erreur`.

### Comment `/answer` sait quoi faire

Il inspecte l'état du `thread_id` :
- **thread interrompu** (champ manquant ou doublon en attente) → il **reprend** le
  graphe avec la réponse (`Command(resume=...)`) ;
- **thread terminé** → il traite `answer` comme une **question de suivi (Q&A)**.

## Exercer le flux (exemples `curl`)

```bash
# 1) Analyser une facture
curl -s -F "user_id=u1" -F "activite=influenceur BNC" \
     -F "file=@facture.pdf" http://localhost:8000/analyze

# -> si status == en_attente_utilisateur, récupérez thread_id + pending.question

# 2) Répondre à un champ manquant / confirmer un doublon
curl -s -X POST http://localhost:8000/answer \
     -H "Content-Type: application/json" \
     -d '{"thread_id":"<THREAD>","answer":"1234,56"}'

# 3) Poser une question de suivi (une fois la facture traitée)
curl -s -X POST http://localhost:8000/answer \
     -H "Content-Type: application/json" \
     -d '{"thread_id":"<THREAD>","answer":"Cette facture est-elle déductible ?"}'

# 4) Lister les factures
curl -s http://localhost:8000/invoices/u1
```

## Démo & tests (aucun service externe)

Les deux utilisent un **client Mistral simulé** + **mongomock** + checkpointer
en mémoire : ni clé API ni MongoDB ne sont nécessaires.

```bash
python -m examples.run_demo     # imprime les 4 scénarios de bout en bout
pytest -q                       # suite de tests
```

La suite couvre : chemin nominal + sauvegarde, **interruption champ manquant en
français + reprise**, **confirmation de doublon** (jamais de rejet auto),
traduction d'une facture anglaise, Q&A ancrée, et un garde-fou vérifiant
qu'aucun terme RAG (`chroma|embedding|vectorstore|faiss`) n'apparaît dans `app/`.

## Modèle de données (MongoDB)

- **`invoices`** — facture extraite + analyse + classification + déductibilité.
  Index **UNIQUE** sur `(invoice_number, issuer_tax_id, total_ttc, issue_date)`,
  index sur `user_id`.
- **`chat_sessions`** — historique par `(user_id, document_id)`.

## Gestion d'erreurs

- Échec/rate-limit/timeout Mistral : retries exponentiels puis `MistralError`/`OCRError`.
- Réponse JSON malformée du modèle : requalifiée en erreur.
- Fichier vide/illisible : `status = erreur`.
- Doublon (index unique) au moment de l'insertion : traité comme doublon, non inséré.
Toute exception à la frontière API est renvoyée en `status = erreur`.

## Structure

```
app/
  api.py            # routes FastAPI (lifespan: Mistral + Mongo + graphe)
  graph.py          # assemblage LangGraph + checkpointer MongoDB
  nodes.py          # un nœud par étape + routage + interruptions + Q&A
  schemas.py        # Pydantic : Invoice, GraphState, contrats API
  db.py             # MongoDB : index, CRUD, mémoire de conversation
  mistral_client.py # wrappers OCR + chat, sélection de modèle, retries
  prompts.py        # tous les prompts, en français, centralisés
  config.py         # chargement .env + constantes
tests/              # fakes.py (doublures) + test_flow.py
examples/           # run_demo.py
```

## Notes de conception

- **Aucune décision critique par LLM seul** : l'ordre des étapes, la clé de
  déduplication et l'insertion sont déterministes ; les champs manquants et les
  doublons passent par une validation humaine.
- Les deux points délicats sont commentés dans le code : l'appel **Mistral OCR**
  (`mistral_client.py`) et l'**interruption LangGraph** (`nodes.py`).
- Périmètre volontairement restreint : type `facture` uniquement, pas de RAG,
  pas d'A2A.
```
