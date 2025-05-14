from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from agent import assistant, AgentState
from tools import extract_text, divide


def build_graph():
    """Constructs and compiles the LangGraph state graph."""
    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode([extract_text, divide]))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    compiled_graph = builder.compile()
    compiled_graph.get_graph().draw_mermaid_png(output_file_path='./compiled_graph.png')

    return compiled_graph