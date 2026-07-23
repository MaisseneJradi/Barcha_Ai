"""Démo exécutable de bout en bout — SANS clé Mistral ni MongoDB réel.

Lance les quatre scénarios clés avec un client Mistral simulé et une base
`mongomock` en mémoire, et imprime le déroulé :
  1. Facture complète  -> analyse + classification + déductibilité + sauvegarde
  2. Champ manquant     -> interruption (question FR) puis reprise
  3. Doublon            -> interruption de confirmation puis insertion ignorée
  4. Question de suivi  -> Q&A ancrée sur l'OCR stocké + historique

Exécution (depuis la racine du projet) :   python -m examples.run_demo
"""
from __future__ import annotations

import copy
import os
import sys

# Permet `python examples/run_demo.py` en plus de `python -m examples.run_demo`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.types import Command  # noqa: E402

from app.nodes import answer_question  # noqa: E402
from tests.fakes import (  # noqa: E402
    COMPLETE_INVOICE,
    FakeMistral,
    initial_state,
    inspect,
    make_db,
    make_graph,
    run,
)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    db = make_db()

    # 1) Facture complète -----------------------------------------------------
    section("1) Facture complète")
    graph, deps = make_graph(FakeMistral(invoice=copy.deepcopy(COMPLETE_INVOICE)), db)
    cfg = run(graph, initial_state("demo"), "demo-1")
    values, _ = inspect(graph, cfg)
    print("Statut       :", values["status"])
    print("Catégorie    :", values["expense_category"])
    print("Déductible   :", values["deductible"], "—", values["deductibility_reason"])
    print("Analyse      :", values["analysis"])
    print("Sauvegardée  :", values["saved"])

    # 2) Champ manquant -------------------------------------------------------
    section("2) Champ manquant (Total TTC) -> HITL")
    inv = copy.deepcopy(COMPLETE_INVOICE)
    inv["total_ttc"] = None
    graph2, _ = make_graph(FakeMistral(invoice=inv), db)
    cfg2 = run(graph2, initial_state("demo"), "demo-2")
    _, ints = inspect(graph2, cfg2)
    print("Question posée :", ints[0].value["question"])
    print("... l'utilisateur répond « 1234,56 »")
    run(graph2, Command(resume="1234,56"), "demo-2")
    values2, _ = inspect(graph2, cfg2)
    print("Total TTC repris :", values2["invoice"]["total_ttc"])
    print("Statut           :", values2["status"])

    # 3) Doublon --------------------------------------------------------------
    section("3) Doublon (même facture que #1) -> confirmation")
    graph3, _ = make_graph(FakeMistral(invoice=copy.deepcopy(COMPLETE_INVOICE)), db)
    cfg3 = run(graph3, initial_state("demo"), "demo-3")
    _, ints3 = inspect(graph3, cfg3)
    print("Interruption :", ints3[0].value["question"])
    print("Existant     :", ints3[0].value["existing_invoice"]["invoice_number"])
    print("... l'utilisateur confirme « oui »")
    run(graph3, Command(resume="oui"), "demo-3")
    values3, _ = inspect(graph3, cfg3)
    print("Doublon ignoré :", values3["duplicate_skipped"], "| sauvegardée :", values3["saved"])
    print("Nb factures en base pour 'demo' :", len(db.list_invoices("demo")))

    # 4) Q&A ------------------------------------------------------------------
    section("4) Question de suivi (Q&A, sans RAG)")
    doc_id = db.list_invoices("demo")[0]["document_id"]
    ans = answer_question(deps, "demo", doc_id, "Cette facture est-elle déductible ?")
    print("Q: Cette facture est-elle déductible ?")
    print("R:", ans)


if __name__ == "__main__":
    main()
