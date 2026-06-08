from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from state import IPLAgentState

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def records_node(state: IPLAgentState) -> IPLAgentState:
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="ipl_rag"
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4, "filter": {"section": "records"}}
    )

    docs = retriever.invoke(state["user_query"])
    print(f"[Records] retrieved {len(docs)} chunks")

    return {**state, "retrieved_chunks": docs}