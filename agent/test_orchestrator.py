"""
Test suite for the self-evaluating RAG orchestrator.

Tests 6 scenarios:
1. Normal direct question
2. Question where query rewrite/expansion is useful
3. Failed retrieval -> diagnosis -> different retrieval strategy
4. Failed faithfulness -> stricter generation
5. Successful evaluation -> finalize
6. Max iterations -> fallback

Each iteration logs:
- original query
- query actually used
- tools used
- number of retrieved documents
- evaluation scores
- failed metrics
- diagnosis/retry action
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import graph


# ============================================================
# Iteration logger
# ============================================================

def log_iteration(iteration_num, state, action=""):
    tools = state.get("tools_used", [])
    chunks = state.get("retrieved_chunks", [])
    scores = state.get("scores", {})
    failed = state.get("failed_metrics", [])
    used_query = state.get("used_query", state.get("query", ""))
    original_query = state.get("query", "")

    print(f"  --- Iteration {iteration_num} ---")
    print(f"  Original query:   {original_query}")
    print(f"  Query used:       {used_query}")
    print(f"  Tools used:       {tools if tools else '(none yet)'}")
    print(f"  Retrieved docs:   {len(chunks)}")
    if scores:
        print(f"  Scores:           {json.dumps(scores, indent=2)}")
    if failed:
        print(f"  Failed metrics:   {failed}")
    if action:
        print(f"  Diagnosis action: {action}")
    print()


def run_test(test_name, query, max_iterations=3):
    print("=" * 70)
    print(f"TEST: {test_name}")
    print(f"Query: {query}")
    print("=" * 70)

    initial_state = {
        "query": query,
        "max_iterations": max_iterations,
    }

    final_state = graph.invoke(initial_state)

    # Log each attempt from history
    for attempt in final_state.get("attempt_history", []):
        print(f"  [History] Iteration {attempt['iteration']}:")
        print(f"    Tools: {attempt['tools_used']}")
        print(f"    Docs:  {len(attempt['retrieved_chunks'])}")
        print(f"    Scores: {json.dumps(attempt['scores'], indent=2)}")
        print(f"    Avg:   {attempt['avg_score']:.3f}")
        used_q = attempt.get('used_query', final_state.get('query', ''))
        print(f"    Query used: {used_q}")
        print()

    # Log final result
    print("  RESULT:")
    print(f"    Status:     {final_state['status']}")
    print(f"    Confidence: {final_state['confidence']}")
    print(f"    Iterations: {final_state['iteration']}")
    print(f"    Used query: {final_state.get('used_query', 'N/A')}")
    print(f"    Expanded:   {final_state.get('expanded_query', 'N/A')}")
    print(f"    Tools used: {final_state.get('tools_used', [])}")
    print(f"    Docs:       {len(final_state.get('retrieved_chunks', []))}")
    answer = final_state['final_answer']
    if isinstance(answer, str):
        print(f"    Final answer: {answer[:300]}")
    else:
        print(f"    Final answer: {str(answer)[:300]}")
    print()
    print()

    return final_state


# ============================================================
# TEST 1: Normal direct question
# ============================================================

def test_normal_direct():
    return run_test(
        "1. Normal Direct Question",
        "Who founded Insurellm?",
        max_iterations=3,
    )


# ============================================================
# TEST 2: Query rewrite/expansion useful (multi-hop)
# ============================================================

def test_query_rewrite_useful():
    return run_test(
        "2. Query Rewrite Useful (Multi-hop)",
        "Four Insurellm products all launched their 'version 1.0' during the first half of 2025. Name all four products, the exact quarter each launched in, and the core capability each version 1.0 introduced.",
        max_iterations=3,
    )


# ============================================================
# TEST 3: Failed retrieval -> diagnosis -> different strategy
# ============================================================

def test_failed_retrieval():
    return run_test(
        "3. Failed Retrieval -> Diagnosis -> Different Strategy",
        "What is the exact salary difference between the two Thompson employees at Insurellm?",
        max_iterations=3,
    )


# ============================================================
# TEST 4: Failed faithfulness -> stricter generation
# ============================================================

def test_failed_faithfulness():
    return run_test(
        "4. Failed Faithfulness -> Stricter Generation",
        "Which Insurellm contract grants exclusive use of a product, and what are the vertical and region?",
        max_iterations=3,
    )


# ============================================================
# TEST 5: Successful evaluation -> finalize
# ============================================================

def test_successful_finalize():
    return run_test(
        "5. Successful Evaluation -> Finalize",
        "What was Insurellm's very first product?",
        max_iterations=3,
    )


# ============================================================
# TEST 6: Max iterations -> fallback
# ============================================================

def test_max_iterations_fallback():
    return run_test(
        "6. Max Iterations -> Fallback",
        "Compare the blockchain roadmaps of Claimllm and Lifellm including exact launch quarters and specific capabilities.",
        max_iterations=2,
    )


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    results = []

    tests = [
        test_normal_direct,
        test_query_rewrite_useful,
        test_failed_retrieval,
        test_failed_faithfulness,
        test_successful_finalize,
        test_max_iterations_fallback,
    ]

    for test_fn in tests:
        try:
            state = test_fn()
            results.append({
                "test": test_fn.__doc__ or test_fn.__name__,
                "status": state["status"],
                "confidence": state["confidence"],
                "iterations": state["iteration"],
                "tools": state.get("tools_used", []),
                "docs": len(state.get("retrieved_chunks", [])),
            })
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({
                "test": test_fn.__doc__ or test_fn.__name__,
                "status": "error",
                "error": str(e),
            })

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for r in results:
        status = r.get('status', 'error')
        conf = r.get('confidence', 'N/A')
        iters = r.get('iterations', 'N/A')
        tools = r.get('tools', [])
        docs = r.get('docs', 0)
        print(f"  {r['test']}: {status} | confidence={conf} | iters={iters} | tools={tools} | docs={docs}")
    print()
