import json
from pathlib import Path
from statistics import mean
from time import time
from langchain_core.documents import Document
from agent.graph import graph
from evaluation.metrics import calculate_all_metrics


BASE_DIR = Path(__file__).resolve().parent

GOLD_SET = BASE_DIR / "eval-agent.jsonl"
OUTPUT_FILE = BASE_DIR / "agent_results2.jsonl"

K = 10


def load_gold_set():
    tests = []

    with open(GOLD_SET, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tests.append(json.loads(line))

    return tests


def run_benchmark():

    tests = load_gold_set()
    results = []

    print(f"Running agent benchmark on {len(tests)} questions...")

    for i, test in enumerate(tests, 1):

        question = test["question"]

        print(f"\n[{i}/{len(tests)}] {question}")

        # Run the complete LangGraph orchestrator
        state = graph.invoke({
            "query": question,
            "max_iterations": 5,
        })
        retrieved_docs = [
    Document(
        page_content=chunk["content"],
        metadata={"source": chunk["source"]}
    )
    for chunk in state.get("retrieved_chunks", [])
]
        # Evaluate final agent output using your existing metrics.py
        metrics = calculate_all_metrics(
            question=question,
            generated_answer=state["final_answer"],
            reference_answer=test["reference_answer"],
            retrieved_docs=retrieved_docs,
            keywords=test["keywords"],
            k=K,
        )

        result = {
            "question": question,
            "category": test["category"],
            "generated_answer": state["final_answer"],
            "confidence": state["confidence"],
            "status": state["status"],
            "iterations": state["iteration"],
            "tools_used": state.get("tools_used", []),
            "used_query": state.get("used_query", question),
            "metrics": metrics,
        }

        results.append(result)

        print(
            f"Overall={metrics['answer']['overall']:.2f}/5 | "
            f"Recall={metrics['retrieval']['recall_at_k']:.3f} | "
            f"nDCG={metrics['retrieval']['ndcg']:.3f} | "
            f"Iterations={state['iteration']}"
        )

    # Save every question's result
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        for result in results:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + "\n"
            )

    print(f"\nSaved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    run_benchmark()