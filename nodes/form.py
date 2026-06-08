from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from state import IPLAgentState

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def form_node(state: IPLAgentState) -> IPLAgentState:
    # skip if this is a plain venue query with no players/teams to look up form for
    qt = state.get("query_type", "")
    entities = state.get("entities", [])

    if qt == "venue" and not entities:
        print(f"[Form] skipped — plain venue query")
        return {**state, "form_context": []}

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="ipl_rag"
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {
                "$and": [
                    {"section": {"$eq": "form"}},
                    {"season": {"$eq": 2024}}
                ]
            }
        }
    )

    query = f"recent form last 5 matches {', '.join(entities)}" if entities else state["user_query"]
    docs = retriever.invoke(query)
    print(f"[Form] retrieved {len(docs)} chunks")

    return {**state, "form_context": docs}