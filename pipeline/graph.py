from langgraph.graph import END, StateGraph

from pipeline.generate import generate
from pipeline.retrieve import retrieve
from pipeline.state import QueryState


def build_graph():
    graph = StateGraph(QueryState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run_query(question: str) -> QueryState:
    app = build_graph()
    return app.invoke({"question": question})
