import math
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

print("Gemini key loaded:", bool(os.getenv("GOOGLE_API_KEY")))
# =========================
# Retrieval Metrics
# =========================

def relevant_documents(retrieved_docs, keywords):
    keywords = [k.lower() for k in keywords]
    return [
        any(k in doc.page_content.lower() for k in keywords)
        for doc in retrieved_docs
    ]


def calculate_mrr(retrieved_docs, keywords):
    relevance = relevant_documents(retrieved_docs, keywords)

    for rank, relevant in enumerate(relevance, start=1):
        if relevant:
            return 1.0 / rank

    return 0.0


def calculate_dcg(relevances):
    return sum(
        relevance / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def calculate_ndcg(retrieved_docs, keywords, k=10):
    relevance = [
        int(x)
        for x in relevant_documents(retrieved_docs[:k], keywords)
    ]

    dcg = calculate_dcg(relevance)
    idcg = calculate_dcg(sorted(relevance, reverse=True))

    return dcg / idcg if idcg > 0 else 0.0


def calculate_recall_at_k(retrieved_docs, keywords, k=10):
    keywords = [k.lower() for k in keywords]

    if not keywords:
        return 0.0

    text = " ".join(
        doc.page_content.lower()
        for doc in retrieved_docs[:k]
    )

    found = sum(k in text for k in keywords)

    return found / len(keywords)


def calculate_precision_at_k(retrieved_docs, keywords, k=10):
    top_k = retrieved_docs[:k]

    if not top_k:
        return 0.0

    relevance = relevant_documents(top_k, keywords)

    return sum(relevance) / len(top_k)


def calculate_keyword_coverage(retrieved_docs, keywords, k=10):
    return calculate_recall_at_k(
        retrieved_docs,
        keywords,
        k
    ) * 100


def calculate_hit_rate_at_k(retrieved_docs, keywords, k=10):
    relevance = relevant_documents(
        retrieved_docs[:k],
        keywords
    )

    return 1.0 if any(relevance) else 0.0


def calculate_retrieval_metrics(
    retrieved_docs,
    keywords,
    k=10
):
    return {
        "mrr": calculate_mrr(
            retrieved_docs,
            keywords
        ),
        "ndcg": calculate_ndcg(
            retrieved_docs,
            keywords,
            k
        ),
        "recall_at_k": calculate_recall_at_k(
            retrieved_docs,
            keywords,
            k
        ),
        "precision_at_k": calculate_precision_at_k(
            retrieved_docs,
            keywords,
            k
        ),
        "keyword_coverage": calculate_keyword_coverage(
            retrieved_docs,
            keywords,
            k
        ),
        "hit_rate_at_k": calculate_hit_rate_at_k(
            retrieved_docs,
            keywords,
            k
        ),
    }


# =========================
# LLM-as-a-Judge
# =========================

class AnswerEvaluation(BaseModel):

    correctness: float = Field(
        description="Factual correctness, 1 to 5."
    )

    completeness: float = Field(
        description="How completely the answer addresses the question, 1 to 5."
    )

    relevance: float = Field(
        description="How directly the answer addresses the question, 1 to 5."
    )

    faithfulness: float = Field(
        description="How well the answer is supported by retrieved context, 1 to 5."
    )

    overall: float = Field(
        description="Overall answer quality, 1 to 5."
    )

    feedback: str = Field(
        description="Concise explanation of the scores."
    )


def evaluate_answer_with_llm(
    question,
    generated_answer,
    reference_answer,
    retrieved_docs,
    model="gemini-3.1-flash-lite"
):

    context = "\n\n".join(
        f"--- Document {i} ---\n{doc.page_content}"
        for i, doc in enumerate(
            retrieved_docs,
            start=1
        )
    )

    prompt = '''
You are a strict evaluator for a Retrieval-Augmented Generation system.

Evaluate the generated answer using the question, reference answer,
and retrieved context below.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

GENERATED ANSWER:
{generated_answer}

RETRIEVED CONTEXT:
{context}

Score each dimension from 1 to 5:

Correctness:
Is the answer factually correct compared with the reference answer?

Completeness:
Does it answer all important parts of the question?

Relevance:
Does it directly answer the question without unnecessary information?

Faithfulness:
Are the important claims supported by the retrieved context?

Overall:
Give an overall assessment.

Scoring:
1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Be strict. Do not give 5 unless the answer is genuinely excellent.
Reduce correctness for factual errors and faithfulness for unsupported claims.
Give concise feedback.
'''.format(
        question=question,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
        context=context
    )

    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0
    )

    judge = llm.with_structured_output(
        AnswerEvaluation
    )

    return judge.invoke(prompt)


# =========================
# Complete Evaluation
# =========================

def calculate_all_metrics(
    question,
    generated_answer,
    reference_answer,
    retrieved_docs,
    keywords,
    k=10,
    judge_model="gemini-3.1-flash-lite"
):

    retrieval = calculate_retrieval_metrics(
        retrieved_docs,
        keywords,
        k
    )

    answer = evaluate_answer_with_llm(
        question,
        generated_answer,
        reference_answer,
        retrieved_docs,
        judge_model
    )

    return {
        "retrieval": retrieval,
        "answer": answer.model_dump()
    }
