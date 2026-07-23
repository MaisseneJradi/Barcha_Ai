"""Assemblage du graphe LangGraph de l'Agent 3 (dans l'ordre du cahier des charges) :

    ocr -> detect_language -> (translate_to_fr?) -> extract_fields
        -> (ask_missing_field: interrupt + boucle) -> write_analysis
        -> classify_expense -> assess_deductibility -> check_duplicate
        -> (interrupt si doublon) -> save_to_db

Le checkpointer permet aux runs interrompus (HITL) de survivre entre deux
requêtes HTTP : `/analyze` démarre un thread, `/answer` le reprend.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from . import nodes
from .nodes import Deps
from .schemas import GraphState


def _bind(fn, deps: Deps):
    """Lie les dépendances à un nœud via une CLÔTURE à un seul argument.

    Important : on n'utilise pas `functools.partial(..., deps=deps)` +
    `update_wrapper`, car cela poserait `__wrapped__` et `inspect.signature`
    (suivi par LangGraph) remonterait à la signature `(state, deps)` d'origine —
    LangGraph tenterait alors d'injecter `config` dans le paramètre `deps`. Une
    clôture expose une signature propre `(state)`.
    """

    def node(state):
        return fn(state, deps)

    node.__name__ = getattr(fn, "__name__", "node")
    return node


def build_graph(deps: Deps, checkpointer):
    """Construit et compile le graphe principal avec un checkpointer donné."""
    g = StateGraph(GraphState)

    g.add_node("ocr", _bind(nodes.ocr_node, deps))
    g.add_node("detect_language", _bind(nodes.detect_language_node, deps))
    g.add_node("translate_to_fr", _bind(nodes.translate_to_fr_node, deps))
    g.add_node("extract_fields", _bind(nodes.extract_fields_node, deps))
    g.add_node("ask_missing_field", _bind(nodes.ask_missing_field_node, deps))
    g.add_node("write_analysis", _bind(nodes.write_analysis_node, deps))
    g.add_node("classify_expense", _bind(nodes.classify_expense_node, deps))
    g.add_node("assess_deductibility", _bind(nodes.assess_deductibility_node, deps))
    g.add_node("check_duplicate", _bind(nodes.check_duplicate_node, deps))
    g.add_node("save_to_db", _bind(nodes.save_to_db_node, deps))

    g.add_edge(START, "ocr")
    g.add_edge("ocr", "detect_language")
    # fr -> extract_fields ; sinon -> translate_to_fr
    g.add_conditional_edges(
        "detect_language",
        nodes.route_after_detect,
        {"extract_fields": "extract_fields", "translate_to_fr": "translate_to_fr"},
    )
    g.add_edge("translate_to_fr", "extract_fields")
    # champs manquants -> boucle HITL ; sinon -> analyse
    g.add_conditional_edges(
        "extract_fields",
        nodes.route_after_extract,
        {"ask_missing_field": "ask_missing_field", "write_analysis": "write_analysis"},
    )
    g.add_conditional_edges(
        "ask_missing_field",
        nodes.route_after_ask,
        {"ask_missing_field": "ask_missing_field", "write_analysis": "write_analysis"},
    )
    g.add_edge("write_analysis", "classify_expense")
    g.add_edge("classify_expense", "assess_deductibility")
    g.add_edge("assess_deductibility", "check_duplicate")
    g.add_edge("check_duplicate", "save_to_db")
    g.add_edge("save_to_db", END)

    return g.compile(checkpointer=checkpointer)
