import json
import re
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from state import IPLAgentState

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def router_node(state: IPLAgentState) -> IPLAgentState:
    query = state["user_query"]

    prompt = f"""You are an IPL query classifier.

Classify this query and extract entity names (player or team names).

Query: "{query}"

Return ONLY a valid JSON object like this:
{{
  "query_type": "batting",
  "entities": ["Virat Kohli"]
}}

query_type must be one of:
- batting       → stats about a batsman (runs, average, strike rate)
- bowling       → stats about a bowler (wickets, economy)
- team          → captain, coach, home ground, titles of a team
- h2h           → head to head between two teams
- venue         → pitch or ground related
- form          → recent form last 5 matches
- records       → IPL all-time records, milestones, OR when two different numbers
                  are mentioned for the same stat (conflict or verification questions)
- prediction    → who will win a match
- dream11       → fantasy team suggestion
- out_of_scope  → nothing to do with IPL data

Return ONLY the JSON. No explanation."""

    response = llm.invoke(prompt)
    text = response.content.strip()

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        data = json.loads(match.group())
    else:
        data = {"query_type": "out_of_scope", "entities": []}

    print(f"[Router] query_type={data.get('query_type')} | entities={data.get('entities')}")

    return {
        **state,
        "query_type": data.get("query_type", "out_of_scope"),
        "entities": data.get("entities", [])
    }