from langgraph.graph import StateGraph, START, END

from agent.state import RAGState
from agent.nodes import (
    planner_node,
    query_expansion_node,
    retrieval_node,
    generation_node,
    evaluator_node,
    diagnosis_node,
    fallback_node,
    finalize_node,
    direct_response_node,
    route_after_evaluation,
    route_after_planner,
)

builder = StateGraph(RAGState)

builder.add_node("planner", planner_node)
builder.add_node("direct_response", direct_response_node)
builder.add_node("retrieval", retrieval_node)
builder.add_node("generation", generation_node)
builder.add_node("evaluator", evaluator_node)
builder.add_node("diagnosis", diagnosis_node)
builder.add_node("fallback", fallback_node)
builder.add_node("finalize", finalize_node)
builder.add_node("query_expansion", query_expansion_node)

builder.add_edge(START, "planner")
builder.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "retrieve": "query_expansion",
        "no_retrieval": "direct_response",
    },
)

builder.add_edge("query_expansion", "retrieval")
builder.add_edge("retrieval", "generation")
builder.add_edge("generation", "evaluator")

builder.add_conditional_edges(
    "evaluator",
    route_after_evaluation,
    {
        "finalize": "finalize",
        "diagnosis": "diagnosis",
        "fallback": "fallback",
    },
)

builder.add_edge("diagnosis", "retrieval")   # the actual retry loop
builder.add_edge("direct_response", END)
builder.add_edge("finalize", END)
builder.add_edge("fallback", END)

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke({
        "query": "How did Nova Robotics' R&D spending change from 2022 to 2024?",
        "max_iterations": 5,
    })

    print(result["final_answer"])
    print("Confidence:", result["confidence"])
    print("Iterations used:", result["iteration"])
    print("Tools used (last pass):", result["tools_used"])
