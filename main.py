"""
main.py  —  FastAPI entry point
Updated with: session-based conversational memory, answer caching,
and new response fields (confidence_score, used_web_fallback, cache_hit).
"""

import uuid
from contextlib import asynccontextmanager
from typing import Optional
import os
import uvicorn
from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from vectorstore.setup import build_vectorstore
from graph import build_graph
from state import IPLAgentState
from memory import conversation_memory, answer_cache

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    if not os.path.exists("./chroma_db"):
        print("Building vector store...")
        build_vectorstore()
    graph = build_graph()
    print("IPL Intelligence Assistant ready.")
    yield


app = FastAPI(
    title="IPL Intelligence Assistant",
    description="Multi-agent LangGraph RAG system for IPL queries",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = None


# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None   # pass a session_id for multi-turn memory


class QueryResponse(BaseModel):
    answer: str
    query_type: str
    sources: list
    conflict_detected: bool
    nodes_activated: list
    session_id: str                     # returned so client can reuse it
    # NEW fields
    original_query: str
    rewritten_query: str
    confidence_score: float
    used_web_fallback: bool
    cache_hit: bool


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "graph": "ready" if graph else "not ready",
        "cache_stats": answer_cache.stats(),
        "active_sessions": len(conversation_memory.all_sessions()),
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    # Assign or reuse session
    session_id = req.session_id or str(uuid.uuid4())

    # ── Cache check ────────────────────────────────────────────────────
    cached = answer_cache.get(req.question)
    if cached:
        print(f"[Cache] HIT for: '{req.question}'")
        # Still record the user turn in memory
        conversation_memory.add_turn(session_id, "user", req.question)
        conversation_memory.add_turn(session_id, "assistant", cached.answer)
        return QueryResponse(
            answer=cached.answer,
            query_type=cached.query_type,
            sources=cached.sources,
            conflict_detected=cached.conflict_detected,
            nodes_activated=cached.sources,
            session_id=session_id,
            original_query=req.question,
            rewritten_query=req.question,
            confidence_score=1.0,
            used_web_fallback=False,
            cache_hit=True,
        )

    # ── Load conversation history ──────────────────────────────────────
    chat_history = conversation_memory.get_last_n(session_id, n=4)
    conversation_memory.add_turn(session_id, "user", req.question)

    # ── Build initial state ────────────────────────────────────────────
    initial_state: IPLAgentState = {
        "user_query": req.question,
        "original_query": req.question,
        "query_type": "",
        "entities": [],
        "batting_context": [],
        "bowling_context": [],
        "h2h_context": [],
        "venue_context": [],
        "form_context": [],
        "retrieved_chunks": [],
        "final_answer": "",
        "sources": [],
        "conflict_detected": False,
        # memory
        "session_id": session_id,
        "chat_history": chat_history,
        # confidence
        "confidence_score": 0.0,
        "used_web_fallback": False,
        # cache
        "cache_hit": False,
    }

    result = graph.invoke(initial_state)

    # ── Record assistant turn in memory ───────────────────────────────
    conversation_memory.add_turn(session_id, "assistant", result["final_answer"])

    # ── Cache the result ───────────────────────────────────────────────
    dedup_sources = list(set(result["sources"]))
    answer_cache.set(
        query=req.question,
        answer=result["final_answer"],
        query_type=result["query_type"],
        sources=dedup_sources,
        conflict_detected=result["conflict_detected"],
    )

    return QueryResponse(
        answer=result["final_answer"],
        query_type=result["query_type"],
        sources=dedup_sources,
        conflict_detected=result["conflict_detected"],
        nodes_activated=dedup_sources,
        session_id=session_id,
        original_query=result.get("original_query", req.question),
        rewritten_query=result.get("user_query", req.question),
        confidence_score=result.get("confidence_score", 0.0),
        used_web_fallback=result.get("used_web_fallback", False),
        cache_hit=False,
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation memory for a session (e.g., user clicks 'New Chat')."""
    conversation_memory.clear(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.delete("/cache")
async def clear_cache():
    """Clear the answer cache (admin use)."""
    answer_cache.clear_all()
    return {"status": "cache cleared"}


@app.post("/eval")
async def run_eval():
    from tools.evaluation.eval_runner import run_evaluation
    return run_evaluation()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)