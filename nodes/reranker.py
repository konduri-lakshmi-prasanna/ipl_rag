"""
nodes/reranker.py  —  Cross-Encoder Reranking Node
After retrieval, reranks all collected documents by relevance to the query
using a Cross-Encoder model. This dramatically improves answer quality.
"""

from typing import List
from langchain_core.documents import Document
from state import IPLAgentState

# Lazy-load the cross-encoder to avoid startup delay
_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("[Reranker] Cross-Encoder model loaded.")
        except Exception as e:
            print(f"[Reranker] Could not load Cross-Encoder: {e}. Falling back to no reranking.")
            _cross_encoder = "unavailable"
    return _cross_encoder


def _rerank_docs(query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    """Score docs with cross-encoder and return top_k most relevant."""
    if not docs:
        return docs

    model = _get_cross_encoder()
    if model == "unavailable" or model is None:
        return docs[:top_k]

    # Create (query, passage) pairs for the cross-encoder
    pairs = [(query, doc.page_content) for doc in docs]

    try:
        scores = model.predict(pairs)
        # Attach scores and sort descending
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        reranked = [doc for _, doc in scored[:top_k]]
        print(f"[Reranker] {len(docs)} docs → top {len(reranked)} after reranking")
        return reranked
    except Exception as e:
        print(f"[Reranker] Scoring failed: {e}. Returning original order.")
        return docs[:top_k]


def reranker_node(state: IPLAgentState) -> IPLAgentState:
    """
    Collects ALL retrieved documents from all context fields,
    reranks them by relevance to the user query, then redistributes
    top results back to the appropriate context fields.
    """
    query = state["user_query"]
    original_query = state.get("original_query", query)
    # Use original query for reranking (less processed = better semantic match)
    rerank_query = original_query

    # Gather all docs with their source field
    all_docs_with_source: List[tuple] = []
    context_fields = [
        "batting_context",
        "bowling_context",
        "h2h_context",
        "venue_context",
        "form_context",
        "retrieved_chunks",
    ]

    for field in context_fields:
        for doc in state.get(field, []):
            all_docs_with_source.append((field, doc))

    if not all_docs_with_source:
        print("[Reranker] No documents to rerank.")
        return state

    all_docs = [doc for _, doc in all_docs_with_source]
    reranked = _rerank_docs(rerank_query, all_docs, top_k=min(10, len(all_docs)))

    # Redistribute reranked docs back to their original fields
    reranked_set = set(id(doc) for doc in reranked)
    new_state = {**state}
    for field in context_fields:
        new_state[field] = [
            doc for doc in state.get(field, [])
            if id(doc) in reranked_set
        ]

    # If a field ended up empty but had docs originally, keep at least 1
    # (ensures partial queries still get some context)
    for field in context_fields:
        if not new_state[field] and state.get(field):
            new_state[field] = state[field][:1]

    return new_state