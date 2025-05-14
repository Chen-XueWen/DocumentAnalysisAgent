import os
import base64
from typing import Optional
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

# Load OpenAI key from env
os.environ.setdefault("OPENAI_API_KEY", "###")

vision_llm = ChatOpenAI(model="gpt-4o")

def extract_text(img_path: str) -> str:
    """Extract text from an image via multimodal GPT-4o."""
    with open(img_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    msg = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Extract all text; return only the text, no explanations."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        )
    ]
    resp = vision_llm.invoke(msg)
    return resp.content.strip()


def divide(a: int, b: int) -> float:
    """Divide two integers."""
    return a / b