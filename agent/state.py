from typing import TypedDict, List, Dict, Literal


class Attempt(TypedDict):
    """Snapshot of one loop iteration, kept for fallback selection."""
    iteration: int
    tools_used: List[str]
    retrieved_chunks: List[dict]
    answer: str
    scores: Dict[str, float]
    avg_score: float


class RAGState(TypedDict):
    # ---- input ----
    query: str

    # ---- planning (set once, by planner_node) ----
    needs_retrieval: bool
    use_query_expansion: bool
    expanded_query: str

    # ---- per-iteration working memory (overwritten each loop pass) ----
    retrieved_chunks: List[dict]
    tools_used: List[str]
    answer: str
    scores: Dict[str, float]
    failed_metrics: List[str]
    used_query: str             # the query actually used for retrieval (original or expanded)
    faithfulness_strict: bool   # when True, generation uses stricter grounding prompt

    # ---- control flow ----
    iteration: int
    max_iterations: int
    next_action: str          # instruction the diagnosis node injects into the retry

    # ---- memory across iterations, used only by fallback ----
    attempt_history: List[Attempt]

    # ---- final output ----
    final_answer: str
    confidence: Literal["High", "Medium", "Low"]
    status: Literal["pending", "passed", "fallback"]