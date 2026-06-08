from typing import TypedDict, List
from langchain_core.documents import Document

class IPLAgentState(TypedDict):
    user_query: str
    query_type: str          # batting | bowling | h2h | venue | form | records | prediction | dream11
    entities: List[str]      # player/team names extracted from query
    batting_context: List[Document]
    bowling_context: List[Document]
    h2h_context: List[Document]
    venue_context: List[Document]
    form_context: List[Document]
    retrieved_chunks: List[Document]
    final_answer: str
    sources: List[str]
    conflict_detected: bool