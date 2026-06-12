"""
nodes/rewrite.py  —  Query Rewriting Node
Rewrites ambiguous or vague queries into clear, retrieval-friendly queries
before they hit the router. This improves retrieval accuracy significantly.
"""

import re
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from state import IPLAgentState

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def rewrite_node(state: IPLAgentState) -> IPLAgentState:
    """
    Rewrites the user's raw query into a clean, specific query
    that retrieval nodes can handle more accurately.

    Examples:
      "How's Kohli doing lately?"     → "Virat Kohli recent form last 5 matches 2024"
      "Compare MI and CSK head to head" → "Mumbai Indians vs Chennai Super Kings head to head record IPL"
      "Who's the best bowler this year?" → "top IPL bowlers 2024 wickets economy rate"
    """
    query = state["user_query"]
    chat_history = state.get("chat_history", [])

    # Build chat history context string for follow-up awareness
    history_str = ""
    if chat_history:
        recent = chat_history[-4:]  # last 2 turns
        history_str = "\n".join(
            [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in recent]
        )
        history_str = f"\nRecent conversation:\n{history_str}\n"

    prompt = f"""You are an IPL query rewriter. Your job is to rewrite vague or ambiguous queries into clear, specific queries that will retrieve the best results from an IPL vector database.

Rules:
- Expand player nicknames to full names (Kohli → Virat Kohli, Rohit → Rohit Sharma, Bumrah → Jasprit Bumrah, Dhoni → MS Dhoni)
- Expand team nicknames (MI → Mumbai Indians, CSK → Chennai Super Kings, RCB → Royal Challengers Bangalore, KKR → Kolkata Knight Riders, DC → Delhi Capitals, PBKS → Punjab Kings, RR → Rajasthan Royals, SRH → Sunrisers Hyderabad, LSG → Lucknow Super Giants, GT → Gujarat Titans)
- Make ambiguous time references explicit (e.g., "lately" → "2024 season", "this year" → "IPL 2024")
- If it's a follow-up question (uses "he", "they", "that team", "same player"), resolve the reference using recent conversation
- Keep it concise and factual
- Return ONLY a JSON object: {{"rewritten_query": "..."}}
{history_str}
Original query: "{query}"

Return ONLY the JSON. No explanation."""

    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            rewritten = data.get("rewritten_query", query)
        else:
            rewritten = query
    except Exception as e:
        print(f"[Rewrite] Error: {e}, using original query")
        rewritten = query

    print(f"[Rewrite] '{query}' → '{rewritten}'")

    return {
        **state,
        "original_query": query,       # preserve original for display
        "user_query": rewritten,        # overwrite with rewritten version
    }