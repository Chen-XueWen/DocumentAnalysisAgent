import os
import asyncio
from langchain_core.messages import HumanMessage
from graph import build_graph
from langfuse.callback import CallbackHandler

os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "###")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "###")
os.environ.setdefault("LANGFUSE_HOST", "https://us.cloud.langfuse.com")

def run_examples():
    graph = build_graph()
    langfuse_handler = CallbackHandler()

    # Example 1: simple computation
    result1 = graph.invoke(
        input={"messages": [HumanMessage(content="Divide 6790 by 5")],
        "input_file": None},
        config={"callbacks": [langfuse_handler]}
    )
    print("Computation result:")
    for msg in result1["messages"]:
        print(msg.content)

    # Example 2: image-based query
    result2 = graph.invoke(
        input={"messages": [HumanMessage(content="According to the note in the image, what items should I buy for dinner menu?")],
        "input_file": "Batman_training_and_meals.png"},
        config={"callbacks": [langfuse_handler]}
    )
    print("Image query result:")
    for msg in result2["messages"]:
        print(msg.content)


if __name__ == "__main__":
    run_examples()
