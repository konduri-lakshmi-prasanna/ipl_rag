from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from state import IPLAgentState

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def team_node(state: IPLAgentState) -> IPLAgentState:
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="ipl_rag"
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3, "filter": {"section": "team"}}
    )

    entities = state.get("entities", [])
    query = f"team profile {', '.join(entities)}" if entities else state["user_query"]

    docs = retriever.invoke(query)
    print(f"[Team] retrieved {len(docs)} chunks")

    return {**state, "retrieved_chunks": docs}