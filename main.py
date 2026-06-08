from vectorstore.setup import build_vectorstore
from graph import build_graph
from state import IPLAgentState
import os

def initialize():
    if not os.path.exists("./chroma_db"):
        print("First run — building vector store...")
        build_vectorstore()
    else:
        print("Vector store already exists, skipping build.")

def run_query(graph, query: str):
    print(f"\n{'='*50}")
    print(f"QUERY: {query}")
    print('='*50)

    initial_state: IPLAgentState = {
        "user_query": query,
        "query_type": "",
        "entities": [],
        "batting_context": [],
        "bowling_context": [],
        "h2h_context": [],
        "venue_context": [],
        "form_context": [],
        "retrieved_chunks": [],
        "final_answer": "",
        "sources": [],
        "conflict_detected": False
    }

    result = graph.invoke(initial_state)

    print(f"\nANSWER:\n{result['final_answer']}")
    print(f"\nSources used: {list(set(result['sources']))}")
    print(f"Conflict detected: {result['conflict_detected']}")
    return result

if __name__ == "__main__":
    initialize()
    graph = build_graph()

    # Easy queries
    run_query(graph, "Who captains Chennai Super Kings in 2024?")
    run_query(graph, "What is Virat Kohli's career IPL run tally?")
    run_query(graph, "Which venue has the highest average first innings score?")

    # Medium queries
    run_query(graph, "List all bowlers with an economy rate below 7.0")
    run_query(graph, "What is Virat Kohli's form in the last 5 matches?")

    # Hard queries
    run_query(graph, "Who will win if MI plays CSK?")
    run_query(graph, "Is Virat Kohli's career runs 7263 or 7084?")
    run_query(graph, "Suggest a Dream11 XI for MI vs SRH at Wankhede")