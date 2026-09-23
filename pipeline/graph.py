from langgraph.graph import END, StateGraph

from pipeline.generate import generate
from pipeline.grounding_check import grounding_check
from pipeline.grounding_gate import grounding_gate
from pipeline.retrieve import retrieve
from pipeline.state import QueryState


def _route_after_gate(state: QueryState) -> str:
    return "generate" if state.get("gate_passed") else END


def build_graph():
    graph = StateGraph(QueryState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grounding_gate", grounding_gate)
    graph.add_node("generate", generate)
    graph.add_node("grounding_check", grounding_check)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grounding_gate")
    graph.add_conditional_edges("grounding_gate", _route_after_gate, {"generate": "generate", END: END})
    graph.add_edge("generate", "grounding_check")
    graph.add_edge("grounding_check", END)
    return graph.compile()


def run_query(question: str, history: list[dict] | None = None) -> QueryState:
    app = build_graph()
    return app.invoke({"question": question, "history": history or []})
