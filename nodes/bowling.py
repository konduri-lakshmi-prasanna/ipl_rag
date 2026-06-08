from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from state import IPLAgentState

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def bowling_node(state: IPLAgentState) -> IPLAgentState:
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="ipl_rag"
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5, "filter": {"section": "bowling"}}
    )

    entities = state.get("entities", [])
    query = state["user_query"]
    if entities:
        query = f"bowling stats for {', '.join(entities)}"

    docs = retriever.invoke(query)
    print(f"[Bowling] retrieved {len(docs)} chunks")

    return {**state, "bowling_context": docs}