import sys
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Add project root to path for cross-directory imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

from agent.tools import RETRIEVAL_TOOLS
from retrieval.rewrite import rewrite_query
from evaluation.metrics import evaluate_answer_with_llm, calculate_retrieval_metrics

TOOL_MAP = {t.name: t for t in RETRIEVAL_TOOLS}

# LLM instances
gemini_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)


# ============================================================
# 1. PLANNER — runs once, sets up the whole state
# ============================================================

class PlanOutput(BaseModel):
    needs_retrieval: bool = Field(
        description="Whether the question requires document retrieval."
    )

    use_query_expansion: bool = Field(
        description=(
            "Whether the query should be rewritten or expanded before "
            "retrieval. Use for ambiguous, complex, multi-hop, or poorly "
            "phrased queries. Do not use for simple direct factual queries."
        )
    )


planner_llm = gemini_llm.with_structured_output(PlanOutput)


_GREETING_PATTERNS = {
    "hi", "hello", "hey", "hiya", "yo", "howdy", "sup", "what's up",
    "good morning", "good afternoon", "good evening", "good night",
    "thanks", "thank you", "bye", "goodbye", "see you",
    "ok", "okay", "sure", "yes", "no", "maybe",
    "help", "what can you do", "who are you",
}


def planner_node(state):
    query = state["query"].strip().lower().rstrip("!.?")

    # Bypass LLM for greetings and trivial queries
    if query in _GREETING_PATTERNS or len(query.split()) <= 2:
        return {
            "needs_retrieval": False,
            "use_query_expansion": False,
            "iteration": 0,
            "max_iterations": state.get("max_iterations", 5),
            "attempt_history": [],
            "next_action": "",
            "status": "pending",
            "used_query": state["query"],
            "faithfulness_strict": False,
        }

    result = planner_llm.invoke([
        SystemMessage(content=(
            "You are a query planner for a document QA system. Decide whether "
            "retrieval is needed, or not"
        )),
        HumanMessage(content=state["query"]),
    ])

    return {
        "needs_retrieval": result.needs_retrieval,
        "use_query_expansion": result.use_query_expansion,
        "iteration": 0,
        "max_iterations": state.get("max_iterations", 5),
        "attempt_history": [],
        "next_action": "",
        "status": "pending",
        "used_query": state["query"],
        "faithfulness_strict": False,
    }


def route_after_planner(state):
    if state.get("needs_retrieval"):
        return "retrieve"
    return "no_retrieval"


# ============================================================
# DIRECT RESPONSE — lightweight path for greetings / non-doc queries
# ============================================================

def direct_response_node(state):
    response = gemini_llm.invoke(
        f"Answer this user message naturally and concisely:\n\n"
        f"{state['query']}"
    )

    content = response.content
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict)
            else str(part)
            for part in content
        )

    return {
        "final_answer": content,
        "confidence": "High",
        "status": "passed",
        "iteration": 0,
        "retrieved_chunks": [],
        "tools_used": [],
        "scores": {},
        "failed_metrics": [],
        "answer": content,
    }


def query_expansion_node(state):
    if not state.get("use_query_expansion", False):
        return {
            "expanded_query": state["query"],
            "used_query": state["query"],
        }

    rewritten = rewrite_query(state["query"])

    return {
        "expanded_query": rewritten,
        "used_query": rewritten,
    }

# ============================================================
# 2. RETRIEVAL — agent picks and calls one or more tools
# ============================================================

retrieval_llm = gemini_llm.bind_tools(RETRIEVAL_TOOLS)


def retrieval_node(state):
    hint = state.get("next_action", "")
    query_for_retrieval = state.get("expanded_query") or state["query"]

    prompt = (
        "Retrieve context to answer this question using the available tools. "
        "Use only one tool.\n"
        f"Question: {query_for_retrieval}"
    )
    if hint:
        prompt += f"\nRetry guidance from the last evaluation: {hint}"

    response = retrieval_llm.invoke(prompt)

    all_chunks, tools_used = [], []
    for call in response.tool_calls:
        tool_fn = TOOL_MAP[call["name"]]
        all_chunks.extend(tool_fn.invoke(call["args"]))
        tools_used.append(call["name"])

    # dedupe by content prefix, since multiple tools often overlap
    seen, deduped = set(), []
    for chunk in all_chunks:
        key = chunk["content"][:200]
        if key not in seen:
            seen.add(key)
            deduped.append(chunk)

    return {
        "retrieved_chunks": deduped,
        "tools_used": tools_used,
        "used_query": query_for_retrieval,
    }


# ============================================================
# 3. GENERATION — answer strictly from retrieved context
# ============================================================

def generation_node(state):
    chunks = state.get("retrieved_chunks", [])

    # No retrieval needed
    if not chunks:
        response = gemini_llm.invoke(
            f"Answer this user query naturally and concisely:\n\n"
            f"{state['query']}"
        )

        content = response.content

        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict)
                else str(part)
                for part in content
            )

        return {"answer": content}

    context = "\n\n".join(
        f"[{i+1}] (source: {c['source']}) {c['content']}"
        for i, c in enumerate(state["retrieved_chunks"])
    )

    is_strict = state.get("faithfulness_strict", False)
    llm = gemini_llm

    if is_strict:
        prompt = (
            "You are a strict, hallucination-free answer generator.\n"
            "Answer the question using ONLY the context below.\n\n"
            "STRICT RULES:\n"
            "- Every factual claim MUST be directly supported by the context. "
            "Cite the source number [N] for each claim.\n"
            "- Do NOT infer, extrapolate, or combine information beyond what is explicitly stated.\n"
            "- If the context does not contain enough information to fully answer, "
            "state exactly what is missing and what IS available.\n"
            "- Preserve exact names, numbers, dates, and technical terms from the context.\n"
            "- Do NOT guess or fill in gaps with general knowledge.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state['query']}\n\n"
            "Answer (with source citations [N] for every claim):"
        )
    else:
        prompt = (
            "Answer the question using ONLY the context below. Cite the source "
            "number for every factual claim, e.g. [1]. If the context does not "
            "contain the answer, say so explicitly instead of guessing. "
            "Preserve important details like names, numbers, and dates from the context.\n\n"
            f"Context:\n{context}\n\nQuestion: {state['query']}"
        )

    response = llm.invoke(prompt)

    # Handle both string and list response formats (Gemini returns list)
    content = response.content
    if isinstance(content, list):
        text_parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in content]
        content = " ".join(text_parts)

    return {"answer": content}


# ============================================================
# 4. EVALUATOR — single LLM-judge call scoring 4 RAGAS-style metrics
# ============================================================

class EvalOutput(BaseModel):
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float


THRESHOLDS = {
    "context_precision": 0.70,
    "context_recall": 0.70,
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
}

eval_llm = gemini_llm.with_structured_output(EvalOutput)


def evaluator_node(state):
    chunks = state.get("retrieved_chunks", [])
    context = "\n\n".join(c["content"] for c in chunks)

    result = eval_llm.invoke([
        SystemMessage(content=(
            "Score this RAG output from 0 to 1 on each metric.\n"
            "- context_precision: how much of the retrieved context is actually relevant\n"
            "- context_recall: whether the context contains everything needed to answer fully\n"
            "- faithfulness: whether every claim in the answer is grounded in the context\n"
            "- answer_relevancy: whether the answer actually addresses the question asked"
        )),
        HumanMessage(content=(
            f"Question: {state['query']}\n\nContext:\n{context}\n\n"
            f"Answer:\n{state['answer']}"
        )),
    ])

    scores = result.model_dump()
    failed = [m for m, v in scores.items() if v < THRESHOLDS[m]]

    return {
        "scores": scores,
        "failed_metrics": failed,
        "used_query": state.get("used_query", state["query"]),
    }


# ============================================================
# 5. DIAGNOSIS — maps failed metrics to a concrete retry instruction
# ============================================================

ACTION_MAP = {
    "context_precision": "Widen retrieval (higher k) then use reranker_tool to cut to the most relevant chunks.",
    "context_recall": "Switch to hybrid_search_tool or multi_query_tool — the context is missing information.",
    "faithfulness": "Do not change retrieval. Tighten grounding — only state what's in the context, cite every claim.",
    "answer_relevancy": "Use query_rewrite_tool — retrieval may be answering a different question than asked.",
}


def diagnosis_node(state):
    attempt = {
        "iteration": state["iteration"],
        "tools_used": state.get("tools_used", []),
        "retrieved_chunks": state.get("retrieved_chunks", []),
        "answer": state["answer"],
        "scores": state["scores"],
        "avg_score": sum(state["scores"].values()) / len(state["scores"]),
        "used_query": state.get("used_query", state["query"]),
    }

    hints = [ACTION_MAP[m] for m in state["failed_metrics"] if m in ACTION_MAP]

    # When faithfulness fails, flag generation to use stricter grounding on next pass
    faithfulness_strict = "faithfulness" in state.get("failed_metrics", [])

    return {
        "attempt_history": state["attempt_history"] + [attempt],
        "iteration": state["iteration"] + 1,
        "next_action": " ".join(hints),
        "faithfulness_strict": faithfulness_strict,
    }


# ============================================================
# 6. FALLBACK — cap hit, return the best-scoring attempt so far
# ============================================================

def fallback_node(state):
    best = max(state["attempt_history"], key=lambda a: a["avg_score"])
    return {"final_answer": best["answer"], "confidence": "Low", "status": "fallback"}


# ============================================================
# 7. FINALIZE — all metrics passed
# ============================================================

def finalize_node(state):
    avg = sum(state["scores"].values()) / len(state["scores"])
    confidence = "High" if avg >= 0.85 else "Medium"
    return {"final_answer": state["answer"], "confidence": confidence, "status": "passed"}


# ============================================================
# ROUTER — conditional edge function after evaluator_node
# ============================================================

def route_after_evaluation(state):
    if not state["failed_metrics"]:
        return "finalize"
    if state["iteration"] >= state["max_iterations"]:
        return "fallback"
    return "diagnosis"
