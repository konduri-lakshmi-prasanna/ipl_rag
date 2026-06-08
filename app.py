import streamlit as st
import os
from vectorstore.setup import build_vectorstore
from graph import build_graph
from state import IPLAgentState

# build vectorstore once
if not os.path.exists("./chroma_db"):
    build_vectorstore()

# load graph once (cached so it doesn't reload every query)
@st.cache_resource
def load_graph():
    return build_graph()

graph = load_graph()

# --- UI ---
st.title("🏏 IPL Intelligence Assistant")
st.markdown("Powered by **LangGraph** multi-agent RAG")

query = st.text_input("Ask anything about IPL:", placeholder="e.g. Who captains CSK in 2024?")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):

        initial_state: IPLAgentState = {
            "user_query": query,
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
            "conflict_detected": False
        }

        result = graph.invoke(initial_state)

    # show answer
    st.markdown("### Answer")
    st.write(result["final_answer"])

    # show metadata
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Nodes activated:**")
        st.write(list(set(result["sources"])))
    with col2:
        st.markdown("**Conflict detected:**")
        if result["conflict_detected"]:
            st.error("⚠️ Yes — data conflict found")
        else:
            st.success("✅ No conflicts")

    # show query type
    st.markdown(f"**Query classified as:** `{result['query_type']}`")