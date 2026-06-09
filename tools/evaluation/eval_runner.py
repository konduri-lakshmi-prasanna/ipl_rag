from graph import build_graph
from state import IPLAgentState

EVAL_QUERIES = [
    # Easy
    ("E1", "What is the minimum CGPA to join RCB squad?", "out_of_scope"),
    ("E2", "Who captains Chennai Super Kings in 2024?", "team"),
    ("E3", "How many IPL titles has Mumbai Indians won?", "team"),
    ("E4", "What is Virat Kohli's career IPL run tally?", "batting"),
    ("E5", "Who has taken the most wickets in IPL history?", "bowling"),
    ("E6", "What is the highest team total in IPL history?", "records"),
    ("E7", "What type of pitch is the Chinnaswamy Stadium known for?", "venue"),
    ("E8", "How many matches has MS Dhoni played in IPL?", "records"),
    # Medium
    ("M1", "Which teams won the IPL title more than once between 2019 and 2024?", "team"),
    ("M2", "List all bowlers with an economy rate below 7.0", "bowling"),
    ("M3", "Which opener has the highest strike rate among batters?", "batting"),
    ("M4", "Compare Jasprit Bumrah and Rashid Khan on all bowling metrics", "bowling"),
    ("M5", "Which venue has the highest average first innings score?", "venue"),
    ("M6", "How many times have MI and CSK played each other?", "h2h"),
    ("M7", "Who won the last 5 matches between KKR and RCB?", "h2h"),
    ("M8", "What is Virat Kohli's form in the last 5 matches?", "form"),
    ("M9", "Which team should bat first at Eden Gardens and why?", "venue"),
    ("M10", "Which player has the most centuries in IPL history?", "records"),
    # Hard
    ("H1", "Suggest a Dream11 XI for KKR vs RCB at Eden Gardens", "dream11"),
    ("H2", "Who is likely to win if RR plays SRH at Wankhede? Justify.", "prediction"),
    ("H3", "Which team has been the most consistent across all seasons from 2019 to 2024?", "team"),
    ("H4", "What bowling strategy should MI use against RCB at Chinnaswamy?", "prediction"),
    ("H5", "Compare Rohit Sharma and KL Rahul as IPL captains", "batting"),
    ("H6", "Has any conflict been detected in Virat Kohli career runs data?", "records"),
    ("H7", "Which is better chasing or defending at Hyderabad? Use historical data.", "prediction"),
    # Expert
    ("X1", "Who should I pick as Dream11 captain for every match this week?", "out_of_scope"),
    ("X2", "What is Kohli average against left arm pace specifically?", "out_of_scope"),
    ("X3", "Is Yuzvendra Chahal wicket count 205 or 187?", "records"),
    ("X4", "Predict the IPL 2025 champion.", "out_of_scope"),
    ("X5", "Which player has the highest salary in IPL 2024 auction?", "out_of_scope"),
    ("X6", "What is the BCCI net worth?", "out_of_scope"),
    ("X7", "Tell me everything about cricket.", "out_of_scope"),
    ("X8", "Is Sachin Tendulkar in this IPL dataset?", "out_of_scope"),
    ("X9", "Who won the Best Batsman award in IPL 2024?", "out_of_scope"),
    ("X10", "Suggest a team for BCCI next T20 World Cup squad.", "out_of_scope"),
]


def run_evaluation():
    graph = build_graph()
    results = []
    passed = 0

    print("\n" + "="*60)
    print("IPL INTELLIGENCE ASSISTANT — EVALUATION")
    print("="*60)

    for qid, query, expected_type in EVAL_QUERIES:
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
            "conflict_detected": False,
        }

        try:
            result = graph.invoke(initial_state)
            actual_type = result.get("query_type", "unknown")
            routed_correctly = (actual_type == expected_type) or (
                expected_type == "out_of_scope" and "not available" in result["final_answer"].lower()
            )
            if routed_correctly:
                passed += 1
            status = "✅ PASS" if routed_correctly else "❌ FAIL"

            print(f"\n[{qid}] {status}")
            print(f"  Query   : {query[:70]}")
            print(f"  Expected: {expected_type} | Got: {actual_type}")
            print(f"  Answer  : {result['final_answer'][:120]}...")

            results.append({
                "id": qid,
                "query": query,
                "expected_type": expected_type,
                "actual_type": actual_type,
                "passed": routed_correctly,
                "answer_snippet": result["final_answer"][:200],
                "conflict": result["conflict_detected"],
                "sources": list(set(result["sources"])),
            })

        except Exception as e:
            print(f"\n[{qid}] ❌ ERROR: {e}")
            results.append({
                "id": qid, "query": query,
                "expected_type": expected_type,
                "actual_type": "error",
                "passed": False,
                "answer_snippet": str(e),
                "conflict": False, "sources": [],
            })

    total = len(EVAL_QUERIES)
    print(f"\n{'='*60}")
    print(f"SCORE: {passed}/{total} ({round(passed/total*100)}%)")
    print(f"{'='*60}\n")

    return {"score": passed, "total": total, "results": results}


if __name__ == "__main__":
    run_evaluation()