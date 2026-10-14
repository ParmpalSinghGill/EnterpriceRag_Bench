"""FastAPI Application for AWS Deployment (ECS / App Runner / EC2).

Provides minimal, production-ready endpoints:
- GET /health: Health check probe
- POST /query: Standard RAG Query Engine (Top-7 context)
- POST /chat: Autonomous ReAct Agent conversation
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llamaindex_app.agent import get_agent, get_query_engine

app = FastAPI(
    title="Enterprise RAG Service (LlamaIndex)",
    description="Production-ready Enterprise RAG & Agent API for AWS deployment",
    version="1.0.0",
)

# Lazy singleton instances
_query_engine = None
_agent = None


def get_cached_query_engine():
    global _query_engine
    if _query_engine is None:
        _query_engine = get_query_engine(top_k=7, use_expansion=True)
    return _query_engine


def get_cached_agent():
    global _agent
    if _agent is None:
        _agent = get_agent(top_k=7, use_expansion=True)
    return _agent


class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's question to answer")
    top_k: Optional[int] = Field(7, description="Number of context documents to retrieve")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]


class ChatRequest(BaseModel):
    message: str = Field(..., description="Message for the conversational ReAct agent")


class ChatResponse(BaseModel):
    message: str
    reply: str


@app.get("/health")
def health() -> Dict[str, str]:
    """AWS target group health check endpoint."""
    return {"status": "healthy", "engine": "llamaindex-hybrid"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest) -> QueryResponse:
    """Execute high-precision hybrid RAG query."""
    try:
        engine = get_cached_query_engine()
        res = engine.query(req.question)

        sources = []
        for node in getattr(res, "source_nodes", []):
            sources.append(
                {
                    "node_id": node.node.node_id,
                    "score": node.score,
                    "metadata": node.node.metadata,
                }
            )

        return QueryResponse(
            question=req.question,
            answer=str(res.response),
            sources=sources,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """Chat with the autonomous ReAct search agent."""
    try:
        agent = get_cached_agent()
        res = await agent.run(user_msg=req.message)
        reply_text = getattr(res, "response", str(res))
        return ChatResponse(
            message=req.message,
            reply=str(reply_text),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("llamaindex_app.app:app", host="0.0.0.0", port=port, reload=False)

