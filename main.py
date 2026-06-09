from contextlib import asynccontextmanager
from typing import Optional
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from vectorstore.setup import build_vectorstore
from graph import build_graph
from state import IPLAgentState

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
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = None


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    query_type: str
    sources: list
    conflict_detected: bool
    nodes_activated: list


@app.get("/health")
async def health():
    return {"status": "ok", "graph": "ready" if graph else "not ready"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    initial_state: IPLAgentState = {
        "user_query": req.question,
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
    }
    result = graph.invoke(initial_state)
    return QueryResponse(
        answer=result["final_answer"],
        query_type=result["query_type"],
        sources=list(set(result["sources"])),
        conflict_detected=result["conflict_detected"],
        nodes_activated=list(set(result["sources"])),
    )


@app.post("/eval")
async def run_eval():
    from evaluation.eval_runner import run_evaluation
    return run_evaluation()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)