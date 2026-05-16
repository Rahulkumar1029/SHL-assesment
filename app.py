# pyrefly: ignore [missing-import]
import os
from dotenv import load_dotenv

# Load env vars FIRST so LangSmith picks them up before any langchain imports
load_dotenv()

# pyrefly: ignore [missing-import]
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List

from langchain_core.messages import HumanMessage, AIMessage
from langsmith import traceable

from src.core.graph import SHLGraphBuilder

graph_instance = None


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class RecommendationItem(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[RecommendationItem]
    end_of_conversation: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph_instance
    graph_instance = SHLGraphBuilder()
    yield


app = FastAPI(
    title="SHL Assessment Agent",
    description="Conversational agent for SHL assessment recommendations",
    version="1.0.0",
    lifespan=lifespan,
)


def convert_messages(messages: List[Message]):
    converted = []
    for msg in messages:
        if msg.role == "assistant":
            converted.append(AIMessage(content=msg.content))
        else:
            # Treat "user", "human", or any unknown role (e.g. Swagger default "string") as HumanMessage
            converted.append(HumanMessage(content=msg.content))
    return converted


@traceable(
    run_type="chain",
    name="shl_chat_endpoint",
    project_name=os.getenv("LANGSMITH_PROJECT", "shl-assessment-agent"),
)
def _run_graph_traced(messages_dicts: list) -> dict:
    """
    Thin wrapper around graph_instance.invoke so LangSmith captures:
    - the raw input message list
    - the full graph execution as a nested trace
    - the structured output (reply, recommendations, end_of_conversation)
    """
    converted = []
    for m in messages_dicts:
        if m["role"] == "assistant":
            converted.append(AIMessage(content=m["content"]))
        else:
            # Treat "user", "human", or unknown roles like Swagger's "string" as HumanMessage
            converted.append(HumanMessage(content=m["content"]))

    result = graph_instance.invoke(converted)

    # Normalize recommendations to plain dicts for serialization
    raw_recs = result.get("recommendations", [])
    recs_out = []
    for rec in raw_recs:
        if isinstance(rec, dict):
            recs_out.append({
                "name": rec.get("name", ""),
                "url": str(rec.get("url", "")),
                "test_type": rec.get("test_type", ""),
            })
        else:
            recs_out.append({
                "name": rec.name,
                "url": str(rec.url),
                "test_type": rec.test_type,
            })

    return {
        "reply": result.get("reply", ""),
        "recommendations": recs_out,
        "end_of_conversation": result.get("end_of_conversation", False),
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Convert to plain dicts so @traceable can serialize the input cleanly
        messages_dicts = [
            {"role": m.role, "content": m.content}
            for m in request.messages
        ]

        result = _run_graph_traced(messages_dicts)

        return ChatResponse(
            reply=result["reply"],
            recommendations=[
                RecommendationItem(
                    name=r["name"],
                    url=r["url"],
                    test_type=r["test_type"],
                )
                for r in result["recommendations"]
            ],
            end_of_conversation=result["end_of_conversation"],
        )

    except Exception as e:
        print(f"[ENDPOINT ERROR] {e}")
        import traceback
        traceback.print_exc()
        return ChatResponse(
            reply="An internal error occurred while processing the request.",
            recommendations=[],
            end_of_conversation=False,
        )