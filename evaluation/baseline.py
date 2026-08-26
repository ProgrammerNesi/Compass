import json
import time
from pathlib import Path

from retrieval.answer import answer_question

BASE_DIR = Path(__file__).parent

GOLD_SET = BASE_DIR / "gold_set.jsonl"

OUTPUT_FILE = BASE_DIR / "baseline_results.jsonl"


def load_gold_set():

    tests = []

    with open(
        GOLD_SET,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():

                tests.append(
                    json.loads(line)
                )

    return tests


def run_baseline():

    tests = load_gold_set()

    results = []

    print(
        f"Running baseline on {len(tests)} questions..."
    )

    for index, test in enumerate(tests, 1):

        question = test["question"]

        print(
            f"\n[{index}/{len(tests)}] {question}"
        )

        answer, docs = answer_question(
            question
        )

        result = {
            "question": question,
            "reference_answer": test[
                "reference_answer"
            ],
            "keywords": test[
                "keywords"
            ],
            "category": test[
                "category"
            ],
            "generated_answer": answer,
            "retrieved_documents": [
                {
                    "source": doc.metadata.get(
                        "source"
                    ),
                    "content": doc.page_content
                }
                for doc in docs
            ]
        }

        results.append(result)

        print("Answer:")
        print(answer)

        # Wait 1.5 minutes after every 7th question
        if index % 7 == 0 and index < len(tests):
            print("\n⏳ Waiting 90 seconds before continuing...")
            time.sleep(90)


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for result in results:

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        f"\nBaseline saved to:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":

    run_baseline()