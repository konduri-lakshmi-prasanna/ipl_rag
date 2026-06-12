import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query for live IPL news or current player info")


def web_search(query: str) -> str:
    """DuckDuckGo search — no API key needed."""
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=8,
        )
        data = response.json()
        if data.get("AbstractText"):
            return data["AbstractText"]
        if data.get("Answer"):
            return data["Answer"]
        topics = data.get("RelatedTopics", [])
        snippets = [t["Text"] for t in topics[:3] if isinstance(t, dict) and t.get("Text")]
        return "\n".join(snippets) if snippets else "No results found."
    except Exception as e:
        return f"Search error: {e}"


web_search_tool = StructuredTool.from_function(
    func=web_search,
    name="web_search",
    description="Search live web for current IPL news, player updates, match results not in the dataset.",
    args_schema=WebSearchInput,
)