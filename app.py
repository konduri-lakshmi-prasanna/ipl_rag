"""
app.py  —  Streamlit UI
Updated with: conversational memory (multi-turn chat), cache hit badge,
confidence score display, web fallback indicator, and query rewrite display.
"""

import uuid
import streamlit as st
import os
from vectorstore.setup import build_vectorstore
from graph import build_graph
from state import IPLAgentState
from nodes.memory import conversation_memory, answer_cache

# ── Build vectorstore once ─────────────────────────────────────────────────
if not os.path.exists("./chroma_db"):
    build_vectorstore()


# ── Load graph once (cached) ───────────────────────────────────────────────
@st.cache_resource
def load_graph():
    return build_graph()

graph = load_graph()


# ── Session state setup ────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {role, content, metadata}


# ── UI Layout ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="IPL Intelligence Assistant", page_icon="🏏", layout="wide")
st.title("🏏 IPL Intelligence Assistant")
st.markdown("Powered by **LangGraph** multi-agent RAG with conversational memory")

# Sidebar: session info + controls
with st.sidebar:
    st.header("Session")
    st.code(f"ID: {st.session_state.session_id[:8]}...", language=None)
    if st.button("🗑️ New Conversation"):
        conversation_memory.clear(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.header("Cache Stats")
    stats = answer_cache.stats()
    st.metric("Cached answers", stats["active_entries"])
    if st.button("Clear cache"):
        answer_cache.clear_all()
        st.success("Cache cleared.")

    st.divider()
    st.markdown("**Query types supported:**")
    st.markdown("""
- 🏏 Batting / Bowling stats
- ⚔️ Head-to-Head records
- 🏟️ Venue / Pitch reports
- 📈 Recent form
- 📜 Records & milestones
- 🏆 Team info
- 🔮 Match prediction
- 🌟 Dream11 suggestions
""")


# ── Chat history display ───────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            cols = st.columns(4)
            with cols[0]:
                st.caption(f"Type: `{meta.get('query_type', 'N/A')}`")
            with cols[1]:
                conf = meta.get("confidence_score", 0)
                color = "🟢" if conf >= 0.7 else "🟡" if conf >= 0.4 else "🔴"
                st.caption(f"{color} Confidence: {conf:.0%}")
            with cols[2]:
                if meta.get("cache_hit"):
                    st.caption("⚡ Cached")
                elif meta.get("used_web_fallback"):
                    st.caption("🌐 Web fallback used")
            with cols[3]:
                if meta.get("conflict_detected"):
                    st.caption("⚠️ Conflict detected")

            # Show query rewrite if it changed
            orig = meta.get("original_query", "")
            rew = meta.get("rewritten_query", "")
            if orig and rew and orig.lower().strip() != rew.lower().strip():
                with st.expander("🔄 Query rewritten"):
                    st.markdown(f"**Original:** {orig}")
                    st.markdown(f"**Rewritten:** {rew}")


# ── Chat input ─────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about IPL..."):

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ── Cache check ────────────────────────────────────────────────────
    cached = answer_cache.get(prompt)
    if cached:
        metadata = {
            "query_type": cached.query_type,
            "confidence_score": 1.0,
            "cache_hit": True,
            "used_web_fallback": False,
            "conflict_detected": cached.conflict_detected,
            "original_query": prompt,
            "rewritten_query": prompt,
        }
        with st.chat_message("assistant"):
            st.write(cached.answer)
            st.caption("⚡ Cached answer")
        st.session_state.messages.append({
            "role": "assistant",
            "content": cached.answer,
            "metadata": metadata,
        })
        # Still update memory
        conversation_memory.add_turn(st.session_state.session_id, "user", prompt)
        conversation_memory.add_turn(st.session_state.session_id, "assistant", cached.answer)
        st.stop()

    # ── Run graph ──────────────────────────────────────────────────────
    chat_history = conversation_memory.get_last_n(st.session_state.session_id, n=4)
    conversation_memory.add_turn(st.session_state.session_id, "user", prompt)

    initial_state: IPLAgentState = {
        "user_query": prompt,
        "original_query": prompt,
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
        "session_id": st.session_state.session_id,
        "chat_history": chat_history,
        "confidence_score": 0.0,
        "used_web_fallback": False,
        "cache_hit": False,
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = graph.invoke(initial_state)

        answer = result["final_answer"]
        st.write(answer)

        # Metadata display
        meta = {
            "query_type": result.get("query_type", "N/A"),
            "confidence_score": result.get("confidence_score", 0.0),
            "cache_hit": False,
            "used_web_fallback": result.get("used_web_fallback", False),
            "conflict_detected": result.get("conflict_detected", False),
            "original_query": result.get("original_query", prompt),
            "rewritten_query": result.get("user_query", prompt),
        }

        cols = st.columns(4)
        with cols[0]:
            st.caption(f"Type: `{meta['query_type']}`")
        with cols[1]:
            conf = meta["confidence_score"]
            color = "🟢" if conf >= 0.7 else "🟡" if conf >= 0.4 else "🔴"
            st.caption(f"{color} Confidence: {conf:.0%}")
        with cols[2]:
            if meta["used_web_fallback"]:
                st.caption("🌐 Web fallback used")
        with cols[3]:
            if meta["conflict_detected"]:
                st.caption("⚠️ Conflict detected")

        orig = meta["original_query"]
        rew = meta["rewritten_query"]
        if orig.lower().strip() != rew.lower().strip():
            with st.expander("🔄 Query rewritten"):
                st.markdown(f"**Original:** {orig}")
                st.markdown(f"**Rewritten:** {rew}")

    # Save to session and memory
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "metadata": meta,
    })
    conversation_memory.add_turn(st.session_state.session_id, "assistant", answer)

    # Cache the result
    answer_cache.set(
        query=prompt,
        answer=answer,
        query_type=result.get("query_type", ""),
        sources=list(set(result.get("sources", []))),
        conflict_detected=result.get("conflict_detected", False),
    )