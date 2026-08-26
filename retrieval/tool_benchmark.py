import json
from pathlib import Path
from statistics import mean

from retrieval.vector import vector_search
from retrieval.bm25 import bm25_search
from retrieval.hybrid import hybrid_search
from retrieval.expansion import mmr_search
from retrieval.reranker import rerank_documents

from evaluation.metrics import calculate_retrieval_metrics


BASE_DIR = Path(__file__).resolve().parent.parent

GOLD_SET = BASE_DIR / "evaluation" / "gold_set.jsonl"
OUTPUT_FILE = BASE_DIR / "evaluation" / "tool_benchmark.json"

K = 10


def load_gold_set():

    tests = []

    with open(GOLD_SET, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():
                tests.append(json.loads(line))

    return tests


def evaluate_tool(name, retrieval_function, tests):

    per_question = []

    print("\n" + "=" * 70)
    print(f"Testing: {name}")
    print("=" * 70)

    for index, test in enumerate(tests, 1):

        question = test["question"]

        print(
            f"[{index}/{len(tests)}] {question}"
        )

        # Run retrieval tool
        retrieved_docs = retrieval_function(question)

        # Evaluate retrieved documents
        metrics = calculate_retrieval_metrics(
            retrieved_docs=retrieved_docs,
            keywords=test["keywords"],
            k=K
        )

        metrics["question"] = question
        metrics["category"] = test["category"]

        per_question.append(metrics)

        print(
            f"  MRR={metrics['mrr']:.3f} | "
            f"nDCG={metrics['ndcg']:.3f} | "
            f"Recall={metrics['recall_at_k']:.3f} | "
            f"Precision={metrics['precision_at_k']:.3f}"
        )

    # Average
    benchmark = {
        "mrr": mean(
            x["mrr"] for x in per_question
        ),

        "ndcg": mean(
            x["ndcg"] for x in per_question
        ),

        "recall_at_k": mean(
            x["recall_at_k"] for x in per_question
        ),

        "precision_at_k": mean(
            x["precision_at_k"] for x in per_question
        ),

        "keyword_coverage": mean(
            x["keyword_coverage"]
            for x in per_question
        ),

        "hit_rate_at_k": mean(
            x["hit_rate_at_k"]
            for x in per_question
        ),

        "per_question": per_question
    }

    return benchmark


def main():

    tests = load_gold_set()

    print(
        f"Running benchmark on {len(tests)} questions..."
    )

    # ------------------------------------------------
    # Define retrieval strategies
    # ------------------------------------------------

    tools = {

        "vector": lambda q:
            vector_search(q, k=K),

        "bm25": lambda q:
            bm25_search(q, k=K),

        "hybrid_rrf": lambda q:
            hybrid_search(q, k=K),

        "mmr": lambda q:
            mmr_search(q, k=K),

        "cross_encoder": lambda q:
            rerank_documents(
                q,
                vector_search(q, k=30),
                k=K
            ),
    }

    # ------------------------------------------------
    # Run benchmark
    # ------------------------------------------------

    results = {}

    for name, function in tools.items():

        results[name] = evaluate_tool(
            name,
            function,
            tests
        )

    # ------------------------------------------------
    # Print comparison
    # ------------------------------------------------

    print("\n\n")
    print("=" * 90)
    print("RETRIEVAL TOOL BENCHMARK")
    print("=" * 90)

    print(
        f"{'Tool':<20}"
        f"{'MRR':<10}"
        f"{'nDCG':<10}"
        f"{'Recall':<10}"
        f"{'Precision':<12}"
        f"{'Hit Rate':<10}"
    )

    print("-" * 90)

    for name, result in results.items():

        print(
            f"{name:<20}"
            f"{result['mrr']:<10.3f}"
            f"{result['ndcg']:<10.3f}"
            f"{result['recall_at_k']:<10.3f}"
            f"{result['precision_at_k']:<12.3f}"
            f"{result['hit_rate_at_k']:<10.3f}"
        )

    # ------------------------------------------------
    # Save
    # ------------------------------------------------

    output = {
        "tests": len(tests),
        "k": K,
        "tools": results
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    print(
        f"\nBenchmark saved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()