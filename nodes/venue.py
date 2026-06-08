from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from state import IPLAgentState

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def venue_node(state: IPLAgentState) -> IPLAgentState:
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="ipl_rag"
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3, "filter": {"section": "venue"}}
    )

    docs = retriever.invoke(state["user_query"])
    print(f"[Venue] retrieved {len(docs)} chunks")

    return {**state, "venue_context": docs}