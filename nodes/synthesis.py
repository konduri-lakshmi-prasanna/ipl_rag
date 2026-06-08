from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv
from state import IPLAgentState
from typing import List

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return "No data available."
    return "\n".join([f"- {doc.page_content}" for doc in docs])

def synthesis_node(state: IPLAgentState) -> IPLAgentState:
    # combine all retrieved context
    context_parts = []

    if state.get("batting_context"):
        context_parts.append("BATTING DATA:\n" + _format_docs(state["batting_context"]))
    if state.get("bowling_context"):
        context_parts.append("BOWLING DATA:\n" + _format_docs(state["bowling_context"]))
    if state.get("h2h_context"):
        context_parts.append("HEAD TO HEAD DATA:\n" + _format_docs(state["h2h_context"]))
    if state.get("venue_context"):
        context_parts.append("VENUE DATA:\n" + _format_docs(state["venue_context"]))
    if state.get("form_context"):
        context_parts.append("RECENT FORM DATA:\n" + _format_docs(state["form_context"]))
    if state.get("retrieved_chunks"):
        context_parts.append("RECORDS DATA:\n" + _format_docs(state["retrieved_chunks"]))

    full_context = "\n\n".join(context_parts) if context_parts else "No context retrieved."

    conflict_warning = ""
    if state.get("conflict_detected"):
        conflict_warning = "\n⚠️ WARNING: Conflicting data detected across sources. Flag this to the user and recommend verification."

    prompt = f"""You are an expert IPL analyst assistant.

Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say 'This information is not available in the dataset.'
{conflict_warning}

CONTEXT:
{full_context}

USER QUESTION: {state["user_query"]}

Give a clear, structured answer:"""

    response = llm.invoke(prompt)
    print(f"[Synthesis] answer generated")

    return {
        **state,
        "final_answer": response.content,
        "sources": [
            doc.metadata.get("section", "unknown")
            for doc in (
                state.get("batting_context", []) +
                state.get("bowling_context", []) +
                state.get("h2h_context", []) +
                state.get("venue_context", []) +
                state.get("form_context", []) +
                state.get("retrieved_chunks", [])
            )
        ]
    }