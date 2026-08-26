import json
from pathlib import Path

from metrics import calculate_all_metrics


BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "baseline_results.jsonl"
OUTPUT_FILE = BASE_DIR / "baseline_metrics.json"

K = 10


def load_results():

    results = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():
                results.append(
                    json.loads(line)
                )

    return results


def run_evaluation():

    results = load_results()

    all_metrics = []

    for index, result in enumerate(
        results,
        start=1
    ):

        # Recreate simple document objects
        # because metrics.py expects doc.page_content
        class Document:

            def __init__(self, content):
                self.page_content = content

        retrieved_docs = [
            Document(doc["content"])
            for doc in result["retrieved_documents"]
        ]

        # Run retrieval + LLM-as-a-judge evaluation
        metrics = calculate_all_metrics(
            question=result["question"],
            generated_answer=result["generated_answer"],
            reference_answer=result["reference_answer"],
            retrieved_docs=retrieved_docs,
            keywords=result["keywords"],
            k=K
        )

        metrics["question"] = result["question"]
        metrics["category"] = result["category"]

        all_metrics.append(metrics)

        retrieval = metrics["retrieval"]
        answer = metrics["answer"]

        print(
            f"[{index}/{len(results)}] "
            f"MRR={retrieval['mrr']:.3f} | "
            f"nDCG={retrieval['ndcg']:.3f} | "
            f"Recall@{K}={retrieval['recall_at_k']:.3f} | "
            f"Precision@{K}={retrieval['precision_at_k']:.3f} | "
            f"LLM Overall={answer['overall']:.2f}/5"
        )

    # Prevent division by zero
    if not all_metrics:
        print("No results found.")
        return

    count = len(all_metrics)

    # =========================
    # Average Retrieval Metrics
    # =========================

    avg_mrr = sum(
        x["retrieval"]["mrr"]
        for x in all_metrics
    ) / count

    avg_ndcg = sum(
        x["retrieval"]["ndcg"]
        for x in all_metrics
    ) / count

    avg_recall = sum(
        x["retrieval"]["recall_at_k"]
        for x in all_metrics
    ) / count

    avg_precision = sum(
        x["retrieval"]["precision_at_k"]
        for x in all_metrics
    ) / count

    avg_coverage = sum(
        x["retrieval"]["keyword_coverage"]
        for x in all_metrics
    ) / count

    avg_hit_rate = sum(
        x["retrieval"]["hit_rate_at_k"]
        for x in all_metrics
    ) / count

    # =========================
    # Average LLM Judge Metrics
    # =========================

    avg_correctness = sum(
        x["answer"]["correctness"]
        for x in all_metrics
    ) / count

    avg_completeness = sum(
        x["answer"]["completeness"]
        for x in all_metrics
    ) / count

    avg_relevance = sum(
        x["answer"]["relevance"]
        for x in all_metrics
    ) / count

    avg_faithfulness = sum(
        x["answer"]["faithfulness"]
        for x in all_metrics
    ) / count

    avg_overall = sum(
        x["answer"]["overall"]
        for x in all_metrics
    ) / count

    # =========================
    # Final Benchmark
    # =========================

    benchmark = {

        "system": "baseline_rag",

        "tests": count,

        "k": K,

        "metrics": {

            "retrieval": {

                "mrr": avg_mrr,

                "ndcg": avg_ndcg,

                "recall_at_k": avg_recall,

                "precision_at_k": avg_precision,

                "keyword_coverage": avg_coverage,

                "hit_rate_at_k": avg_hit_rate
            },

            "answer": {

                "correctness": avg_correctness,

                "completeness": avg_completeness,

                "relevance": avg_relevance,

                "faithfulness": avg_faithfulness,

                "overall": avg_overall
            }
        },

        "per_question": all_metrics
    }

    # Save benchmark

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            benchmark,
            f,
            indent=2
        )

    # =========================
    # Print Benchmark
    # =========================

    print("\n" + "=" * 60)
    print("BASELINE BENCHMARK")
    print("=" * 60)

    print("\nRetrieval Metrics")
    print("-" * 30)

    print(f"MRR:             {avg_mrr:.4f}")
    print(f"nDCG@{K}:         {avg_ndcg:.4f}")
    print(f"Recall@{K}:       {avg_recall:.4f}")
    print(f"Precision@{K}:    {avg_precision:.4f}")
    print(f"Keyword Coverage: {avg_coverage:.2f}%")
    print(f"Hit Rate@{K}:     {avg_hit_rate:.4f}")

    print("\nLLM-as-a-Judge")
    print("-" * 30)

    print(f"Correctness:      {avg_correctness:.2f}/5")
    print(f"Completeness:     {avg_completeness:.2f}/5")
    print(f"Relevance:        {avg_relevance:.2f}/5")
    print(f"Faithfulness:     {avg_faithfulness:.2f}/5")
    print(f"Overall:          {avg_overall:.2f}/5")

    print("\n" + "=" * 60)

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    run_evaluation()