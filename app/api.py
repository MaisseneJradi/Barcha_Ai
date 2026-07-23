"""Couche API FastAPI de l'Agent 3.

Trois routes :
  - POST /analyze          : lance le graphe d'analyse complet.
  - POST /answer           : répond à un champ manquant, confirme un doublon,
                             OU pose une question de suivi (Q&A) selon l'état du thread.
  - GET  /invoices/{user}  : liste les factures d'un utilisateur.

Chaque réponse distingue trois états : completed / en_attente_utilisateur / erreur.
"""
from __future__ import annotations

import asyncio
import base64
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from pymongo import MongoClient

from . import nodes
from .config import get_settings
from .db import Database
from .graph import build_graph
from .mistral_client import MistralClient
from .nodes import Deps
from .schemas import (
    AnalyzeResponse,
    AnswerRequest,
    AnswerResponse,
    Invoice,
    InvoiceListItem,
    PendingQuestion,
    Status,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise les dépendances au démarrage (Mistral, Mongo, graphe)."""
    settings = get_settings()
    settings.require_api_key()

    mongo: MongoClient = MongoClient(settings.mongodb_uri)
    db = Database(mongo, settings.mongodb_db)
    db.ensure_indexes()

    deps = Deps(mistral=MistralClient(settings.mistral_api_key), db=db)
    # Checkpointer MongoDB : réutilise la même instance Mongo (aucune ressource
    # supplémentaire requise au-delà de MONGODB_URI).
    checkpointer = MongoDBSaver(mongo, db_name=settings.checkpoint_db)
    graph = build_graph(deps, checkpointer)

    app.state.deps = deps
    app.state.graph = graph
    try:
        yield
    finally:
        mongo.close()


app = FastAPI(title="LedgerMind — Agent 3 (Analyse documentaire)", lifespan=lifespan)

# Interface web de test, servie en MÊME ORIGINE que l'API (aucun CORS requis).
_FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> str:
    if not _FRONTEND.exists():
        return "<h1>Agent 3</h1><p>frontend/index.html introuvable.</p>"
    return _FRONTEND.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers d'interprétation d'état
# ---------------------------------------------------------------------------
def _interrupts(graph, config) -> Tuple[Any, List[Any]]:
    snap = graph.get_state(config)
    ints = [i for t in snap.tasks for i in (getattr(t, "interrupts", None) or [])]
    return snap, ints


def _build_analyze_response(graph, config, thread_id: str) -> AnalyzeResponse:
    snap, ints = _interrupts(graph, config)
    values: Dict[str, Any] = snap.values or {}
    document_id = values.get("document_id")

    if ints:
        payload = ints[0].value or {}
        pending = PendingQuestion(
            type=payload.get("type"),
            question=payload.get("question", ""),
            field=payload.get("field"),
            existing_invoice=payload.get("existing_invoice"),
            new_invoice=payload.get("new_invoice"),
        )
        return AnalyzeResponse(
            status=Status.en_attente_utilisateur,
            thread_id=thread_id,
            document_id=document_id,
            pending=pending,
        )

    return AnalyzeResponse(
        status=Status.completed,
        thread_id=thread_id,
        document_id=document_id,
        invoice=Invoice(**(values.get("invoice") or {})),
        analysis=values.get("analysis"),
        expense_category=values.get("expense_category"),
        deductible=values.get("deductible"),
        deductibility_reason=values.get("deductibility_reason"),
        saved=values.get("saved"),
        duplicate_skipped=values.get("duplicate_skipped"),
    )


async def _run(graph, payload, config) -> None:
    """Exécute le graphe hors event-loop ; l'interruption HITL n'est pas une erreur."""
    try:
        await asyncio.to_thread(graph.invoke, payload, config)
    except GraphInterrupt:
        # État figé par le checkpointer ; il sera lu via get_state().
        pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    activite: Optional[str] = Form(None),
) -> AnalyzeResponse:
    graph = app.state.graph
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    try:
        data = await file.read()
        if not data:
            return AnalyzeResponse(status=Status.erreur, thread_id=thread_id, error="Fichier vide ou illisible.")
        initial: Dict[str, Any] = {
            "user_id": user_id,
            "document_id": uuid.uuid4().hex,
            "filename": file.filename,
            "mime": file.content_type,
            "file_b64": base64.b64encode(data).decode("ascii"),
            "activite": activite or "",
            "messages": [],
        }
        await _run(graph, initial, config)
        return _build_analyze_response(graph, config, thread_id)
    except Exception as exc:  # noqa: BLE001 - frontière API : on renvoie un statut erreur
        return AnalyzeResponse(status=Status.erreur, thread_id=thread_id, error=str(exc))


@app.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest) -> AnswerResponse:
    graph = app.state.graph
    deps: Deps = app.state.deps
    config = {"configurable": {"thread_id": req.thread_id}}

    try:
        snap, ints = _interrupts(graph, config)
        values: Dict[str, Any] = snap.values or {}

        # Cas 1 : le thread est en attente (champ manquant OU confirmation doublon)
        if ints:
            await _run(graph, Command(resume=req.answer), config)
            analyze_resp = _build_analyze_response(graph, config, req.thread_id)
            return AnswerResponse(
                status=analyze_resp.status,
                thread_id=req.thread_id,
                document_id=analyze_resp.document_id,
                analyze=analyze_resp,
            )

        # Cas 2 : thread terminé -> question de suivi (Q&A) sur la facture
        user_id = values.get("user_id")
        document_id = values.get("document_id")
        if not user_id or not document_id:
            return AnswerResponse(
                status=Status.erreur, thread_id=req.thread_id,
                error="Thread inconnu ou sans facture associée.",
            )
        ans = await asyncio.to_thread(nodes.answer_question, deps, user_id, document_id, req.answer)
        return AnswerResponse(
            status=Status.completed, thread_id=req.thread_id,
            document_id=document_id, answer=ans,
        )
    except Exception as exc:  # noqa: BLE001
        return AnswerResponse(status=Status.erreur, thread_id=req.thread_id, error=str(exc))


@app.get("/invoices/{user_id}", response_model=List[InvoiceListItem])
async def list_invoices(user_id: str) -> List[InvoiceListItem]:
    deps: Deps = app.state.deps
    docs = await asyncio.to_thread(deps.db.list_invoices, user_id)
    out: List[InvoiceListItem] = []
    for d in docs:
        out.append(
            InvoiceListItem(
                document_id=d.get("document_id", ""),
                invoice=Invoice(**(d.get("invoice") or {})),
                analysis=d.get("analysis"),
                expense_category=d.get("expense_category"),
                deductible=d.get("deductible"),
                deductibility_reason=d.get("deductibility_reason"),
                created_at=d.get("created_at"),
            )
        )
    return out
