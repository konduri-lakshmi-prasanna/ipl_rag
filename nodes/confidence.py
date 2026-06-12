"""
nodes/confidence.py  —  Confidence Check + Fallback Web Search Node
Evaluates whether retrieved context is sufficient to answer the query.
If confidence is low, triggers a live web search as fallback.
"""

import re
import json
from typing import List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from state import IPLAgentState
from tools.web_search import web_search

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Minimum number of retrieved doc chunks to consider retrieval "sufficient"
MIN_DOCS_THRESHOLD = 2
# Minimum average doc length (chars) to consider content meaningful
MIN_CONTENT_LENGTH = 50


def _compute_basic_confidence(state: IPLAgentState) -> float:
    """
    Heuristic confidence score (0.0 – 1.0) based on how much context was retrieved.
    0.0 = nothing retrieved, 1.0 = plenty of rich context.
    """
    context_fields = [
        "batting_context", "bowling_context", "h2h_context",
        "venue_context", "form_context", "retrieved_chunks",
    ]
    all_docs: List[Document] = []
    for field in context_fields:
        all_docs.extend(state.get(field, []))

    if not all_docs:
        return 0.0

    # Score by doc count
    count_score = min(len(all_docs) / 5.0, 1.0)  # saturates at 5 docs

    # Score by average content length
    avg_len = sum(len(d.page_content) for d in all_docs) / len(all_docs)
    length_score = min(avg_len / 200.0, 1.0)  # saturates at 200 chars avg

    return round((count_score * 0.6 + length_score * 0.4), 2)


def _llm_confidence_check(query: str, context_preview: str) -> float:
    """
    Ask the LLM to score how well the retrieved context answers the query.
    Returns a float 0.0 – 1.0.
    """
    prompt = f"""You are evaluating whether retrieved context is sufficient to answer a question.

Question: "{query}"

Retrieved context (preview):
{context_preview[:800]}

Rate the context quality from 0 to 10:
- 0-3: Context is missing, irrelevant, or too vague to answer the question
- 4-6: Context partially answers the question
- 7-10: Context clearly and completely answers the question

Return ONLY a JSON: {{"score": <integer 0-10>, "reason": "<one sentence>"}}"""

    try:
        response = llm.invoke(prompt)
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score = int(data.get("score", 5))
            reason = data.get("reason", "")
            print(f"[Confidence] LLM score: {score}/10 — {reason}")
            return round(score / 10.0, 1)
    except Exception as e:
        print(f"[Confidence] LLM check failed: {e}")
    return 0.5  # default neutral


def _run_web_search_fallback(state: IPLAgentState) -> List[Document]:
    """
    Runs DuckDuckGo web search and wraps results as Document objects
    so they integrate cleanly with the existing synthesis pipeline.
    """
    query = state.get("original_query", state["user_query"])
    search_query = f"IPL {query} 2024"
    print(f"[Confidence] Running web fallback for: '{search_query}'")

    result_text = web_search(search_query)

    if not result_text or result_text.startswith("No results") or result_text.startswith("Search error"):
        print("[Confidence] Web search returned no usable results.")
        return []

    # Split into chunks if long
    chunks = [result_text[i:i+400] for i in range(0, len(result_text), 400)]
    docs = [
        Document(
            page_content=chunk,
            metadata={"section": "web_search", "source": "web", "query": search_query}
        )
        for chunk in chunks[:3]  # max 3 web chunks
    ]
    print(f"[Confidence] Web fallback returned {len(docs)} chunks.")
    return docs


def confidence_node(state: IPLAgentState) -> IPLAgentState:
    """
    1. Checks confidence in retrieved context.
    2. If confidence < threshold, runs web search and appends results.
    3. Stores confidence score in state for transparency.
    """
    query = state["user_query"]

    # --- Step 1: Basic heuristic confidence ---
    basic_conf = _compute_basic_confidence(state)
    print(f"[Confidence] Basic heuristic: {basic_conf:.2f}")

    # --- Step 2: LLM-based confidence (only if basic score is borderline) ---
    if 0.2 <= basic_conf <= 0.7:
        all_docs = (
            state.get("batting_context", []) +
            state.get("bowling_context", []) +
            state.get("h2h_context", []) +
            state.get("venue_context", []) +
            state.get("form_context", []) +
            state.get("retrieved_chunks", [])
        )
        preview = "\n".join(d.page_content for d in all_docs[:4])
        llm_conf = _llm_confidence_check(query, preview)
        # Blend the two scores
        final_conf = round((basic_conf * 0.4 + llm_conf * 0.6), 2)
    else:
        final_conf = basic_conf

    print(f"[Confidence] Final confidence score: {final_conf:.2f}")

    # --- Step 3: Trigger web fallback if confidence is low ---
    web_docs = []
    if final_conf < 0.4:
        print(f"[Confidence] Low confidence ({final_conf}). Triggering web fallback.")
        web_docs = _run_web_search_fallback(state)

    # Merge web results into retrieved_chunks
    existing_chunks = state.get("retrieved_chunks", [])
    updated_chunks = existing_chunks + web_docs

    return {
        **state,
        "retrieved_chunks": updated_chunks,
        "confidence_score": final_conf,
        "used_web_fallback": len(web_docs) > 0,
    }