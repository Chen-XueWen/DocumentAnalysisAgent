from typing import TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tools import extract_text, divide

# Bind tools to the LLM
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([divide, extract_text], parallel_tool_calls=False)

class AgentState(TypedDict):
    input_file: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState) -> AgentState:
    """
    The main agent callback: sets up the system message and invokes the LLM or tools.
    """
    desc = (
        "You are an agent that can analyse images and run computations. "
        "Available tools:\n"
        "1) extract_text(img_path: str) -> str\n"
        "2) divide(a: int, b: int) -> float\n"
    )
    sys = SystemMessage(
        content=desc + f"\nCurrent image: {state.get('input_file')}"
    )
    # Invoke LLM (it may call tools under the hood)
    out = llm_with_tools.invoke([sys] + state["messages"])
    return {"messages": [out], "input_file": state.get("input_file")}