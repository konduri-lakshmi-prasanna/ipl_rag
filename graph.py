"""
graph.py  —  LangGraph Workflow Builder
Updated graph with: query rewriting → routing → retrieval →
reranking → confidence check (+ web fallback) → validation → synthesis
"""

from langgraph.graph import StateGraph, END
from state import IPLAgentState
from nodes.rewrite import rewrite_node
from nodes.router import router_node
from nodes.batting import batting_node
from nodes.bowling import bowling_node
from nodes.h2h import h2h_node
from nodes.venue import venue_node
from nodes.form import form_node
from nodes.records import records_node
from nodes.synthesis import synthesis_node
from nodes.validation import validation_node
from nodes.team import team_node
from nodes.reranker import reranker_node
from nodes.confidence import confidence_node


# ─────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────

def route_query(state: IPLAgentState) -> str:
    qt = state.get("query_type", "out_of_scope")

    if qt == "team":                return "team"
    if qt == "batting":             return "batting"
    if qt == "bowling":             return "bowling"
    if qt == "records":             return "records"
    if qt == "venue":               return "venue"
    if qt in ["h2h", "prediction"]: return "h2h"
    if qt == "form":                return "form"
    if qt == "dream11":             return "form"

    return "synthesis"   # out_of_scope — skip retrieval entirely


def route_after_form(state: IPLAgentState) -> str:
    """dream11: form → batting → bowling → venue → reranker"""
    if state.get("query_type") == "dream11":
        return "batting"
    return "reranker"


def route_after_batting(state: IPLAgentState) -> str:
    """dream11: batting → bowling; else → reranker"""
    if state.get("query_type") == "dream11":
        return "bowling"
    return "reranker"


def route_after_bowling(state: IPLAgentState) -> str:
    """dream11: bowling → venue; else → reranker"""
    if state.get("query_type") == "dream11":
        return "venue"
    return "reranker"


def route_after_venue(state: IPLAgentState) -> str:
    """prediction/h2h: venue → form; else → reranker"""
    if state.get("query_type") in ["prediction", "h2h"]:
        return "form"
    return "reranker"


# ─────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(IPLAgentState)

    # ── Register all nodes ──────────────────────────────────────────────
    graph.add_node("rewrite",    rewrite_node)      # NEW: query rewriting
    graph.add_node("router",     router_node)
    graph.add_node("team",       team_node)
    graph.add_node("batting",    batting_node)
    graph.add_node("bowling",    bowling_node)
    graph.add_node("h2h",        h2h_node)
    graph.add_node("venue",      venue_node)
    graph.add_node("form",       form_node)
    graph.add_node("records",    records_node)
    graph.add_node("reranker",   reranker_node)     # NEW: cross-encoder reranking
    graph.add_node("confidence", confidence_node)   # NEW: confidence + web fallback
    graph.add_node("validation", validation_node)
    graph.add_node("synthesis",  synthesis_node)

    # ── Entry point ─────────────────────────────────────────────────────
    # Query is rewritten BEFORE routing for better classification + retrieval
    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "router")

    # ── Routing from router ─────────────────────────────────────────────
    graph.add_conditional_edges("router", route_query, {
        "team":      "team",
        "batting":   "batting",
        "bowling":   "bowling",
        "records":   "records",
        "venue":     "venue",
        "h2h":       "h2h",
        "form":      "form",
        "synthesis": "synthesis",   # out_of_scope: skip straight to synthesis
    })

    # ── Simple nodes → reranker ─────────────────────────────────────────
    # (reranker replaced direct → validation to add reranking step)
    graph.add_edge("team",    "reranker")
    graph.add_edge("records", "reranker")
    graph.add_edge("h2h",     "venue")     # prediction path: h2h → venue → form

    # ── Dream11 multi-hop: form → batting → bowling → venue → reranker ──
    graph.add_conditional_edges("form", route_after_form, {
        "batting": "batting",
        "reranker": "reranker",
    })
    graph.add_conditional_edges("batting", route_after_batting, {
        "bowling":  "bowling",
        "reranker": "reranker",
    })
    graph.add_conditional_edges("bowling", route_after_bowling, {
        "venue":    "venue",
        "reranker": "reranker",
    })
    graph.add_conditional_edges("venue", route_after_venue, {
        "form":     "form",
        "reranker": "reranker",
    })

    # ── Post-retrieval pipeline: reranker → confidence → validation → synthesis ──
    graph.add_edge("reranker",   "confidence")   # NEW
    graph.add_edge("confidence", "validation")   # NEW
    graph.add_edge("validation", "synthesis")
    graph.add_edge("synthesis",  END)

    return graph.compile()