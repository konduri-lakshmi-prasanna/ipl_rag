from langgraph.graph import StateGraph, END
from state import IPLAgentState
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

    return "synthesis"


def route_after_form(state: IPLAgentState) -> str:
    """After form: dream11 continues to batting, everything else stops"""
    if state.get("query_type") == "dream11":
        return "batting"
    return "validation"


def route_after_batting(state: IPLAgentState) -> str:
    """After batting: dream11 continues to bowling, everything else stops"""
    if state.get("query_type") == "dream11":
        return "bowling"
    return "validation"


def route_after_bowling(state: IPLAgentState) -> str:
    """After bowling: dream11 continues to venue, everything else stops"""
    if state.get("query_type") == "dream11":
        return "venue"
    return "validation"


def route_after_venue(state: IPLAgentState) -> str:
    """After venue: prediction continues to form, everything else stops"""
    if state.get("query_type") in ["prediction", "h2h"]:
        return "form"
    return "validation"


def build_graph():
    graph = StateGraph(IPLAgentState)

    # register all nodes
    graph.add_node("router",     router_node)
    graph.add_node("team",       team_node)
    graph.add_node("batting",    batting_node)
    graph.add_node("bowling",    bowling_node)
    graph.add_node("h2h",        h2h_node)
    graph.add_node("venue",      venue_node)
    graph.add_node("form",       form_node)
    graph.add_node("records",    records_node)
    graph.add_node("validation", validation_node)
    graph.add_node("synthesis",  synthesis_node)

    # entry point
    graph.set_entry_point("router")

    # routing from router — one destination per query type
    graph.add_conditional_edges("router", route_query, {
        "team":      "team",
        "batting":   "batting",
        "bowling":   "bowling",
        "records":   "records",
        "venue":     "venue",
        "h2h":       "h2h",
        "form":      "form",
        "synthesis": "synthesis",
    })

    # simple nodes — always go straight to validation
    graph.add_edge("team",    "validation")
    graph.add_edge("records", "validation")

    # batting — dream11 continues to bowling, plain batting stops
    graph.add_conditional_edges("batting", route_after_batting, {
        "bowling":    "bowling",
        "validation": "validation",
    })

    # bowling — dream11 continues to venue, plain bowling stops
    graph.add_conditional_edges("bowling", route_after_bowling, {
        "venue":      "venue",
        "validation": "validation",
    })

    # venue — prediction continues to form, plain venue/dream11 stops
    graph.add_conditional_edges("venue", route_after_venue, {
        "form":       "form",
        "validation": "validation",
    })

    # h2h — always goes to venue next (prediction path)
    graph.add_edge("h2h", "venue")

    # form — dream11 continues to batting, plain form stops
    graph.add_conditional_edges("form", route_after_form, {
        "batting":    "batting",
        "validation": "validation",
    })

    # all paths converge here
    graph.add_edge("validation", "synthesis")
    graph.add_edge("synthesis",  END)

    return graph.compile()