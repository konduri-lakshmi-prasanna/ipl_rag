"""
state.py  —  IPL Agent State
Extended with fields for: conversational memory, confidence scoring,
cache hit tracking, query rewriting, and web fallback status.
"""

from typing import TypedDict, List, Dict, Optional
from langchain_core.documents import Document


class IPLAgentState(TypedDict):
    # ── Core query fields ──────────────────────────────────────────────
    user_query: str                     # current (possibly rewritten) query
    original_query: str                 # raw query before rewriting
    query_type: str                     # batting | bowling | h2h | venue | form |
                                        # records | prediction | dream11 | out_of_scope
    entities: List[str]                 # player/team names extracted from query

    # ── Retrieved context ──────────────────────────────────────────────
    batting_context: List[Document]
    bowling_context: List[Document]
    h2h_context: List[Document]
    venue_context: List[Document]
    form_context: List[Document]
    retrieved_chunks: List[Document]    # records, web fallback results, etc.

    # ── Output ─────────────────────────────────────────────────────────
    final_answer: str
    sources: List[str]
    conflict_detected: bool

    # ── Conversational memory ──────────────────────────────────────────
    session_id: str                     # unique ID per user/session
    chat_history: List[Dict]            # list of {role: str, content: str}

    # ── Confidence & fallback ──────────────────────────────────────────
    confidence_score: float             # 0.0–1.0 confidence in retrieved context
    used_web_fallback: bool             # True if web search was triggered

    # ── Cache ──────────────────────────────────────────────────────────
    cache_hit: bool                     # True if answer came from cache