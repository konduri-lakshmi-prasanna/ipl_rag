from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from state import IPLAgentState

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def batting_node(state: IPLAgentState) -> IPLAgentState:
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="ipl_rag"
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5, "filter": {"section": "batting"}}
    )

    entities = state.get("entities", [])
    query = state["user_query"]
    if entities:
        query = f"batting stats for {', '.join(entities)}"

    docs = retriever.invoke(query)
    print(f"[Batting] retrieved {len(docs)} chunks")

    return {**state, "batting_context": docs}