from state import IPLAgentState

def validation_node(state: IPLAgentState) -> IPLAgentState:
    all_docs = (
        state.get("batting_context", []) +
        state.get("bowling_context", []) +
        state.get("h2h_context", []) +
        state.get("venue_context", []) +
        state.get("form_context", []) +
        state.get("retrieved_chunks", [])
    )

    # only flag conflict if BOTH primary and secondary versions of same entity appear
    conflict_sources = [
        doc.metadata.get("source")
        for doc in all_docs
        if doc.metadata.get("conflict") == True
    ]

    conflict_found = "primary" in conflict_sources and "secondary" in conflict_sources

    print(f"[Validation] conflict_detected={conflict_found}")
    return {**state, "conflict_detected": conflict_found}