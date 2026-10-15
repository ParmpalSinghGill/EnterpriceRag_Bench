"""FastAPI production service powered by Pure LlamaIndex."""

import os
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pure_llamaindex.agent import get_query_engine, get_react_agent

app = FastAPI(
    title="Pure Solo LlamaIndex Enterprise API",
    description="Low-code RAG and Autonomous Agent powered purely by LlamaIndex built-ins.",
    version="1.0.0",
)

_CACHED_QE = None
_CACHED_AGENT = None


def get_cached_engine():
    global _CACHED_QE
    if _CACHED_QE is None:
        _CACHED_QE = get_query_engine(similarity_top_k=7)
    return _CACHED_QE


def get_cached_agent_instance():
    global _CACHED_AGENT
    if _CACHED_AGENT is None:
        _CACHED_AGENT = get_react_agent(similarity_top_k=7)
    return _CACHED_AGENT


class QueryRequest(BaseModel):
    question: str = Field(..., description="Enterprise question to answer")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]] = []


class ChatRequest(BaseModel):
    message: str = Field(..., description="Message for the autonomous ReAct agent")


class ChatResponse(BaseModel):
    message: str
    reply: str


@app.get("/health")
def health():
    return {"status": "ok", "framework": "Pure LlamaIndex"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Fast, single-shot pure LlamaIndex Vector Query."""
    try:
        engine = get_cached_engine()
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
async def chat(req: ChatRequest) -> ChatResponse:
    """Autonomous multi-step ReAct agent workflow."""
    try:
        agent = get_cached_agent_instance()
        res = await agent.run(user_msg=req.message)
        return ChatResponse(
            message=req.message,
            reply=str(res),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("pure_llamaindex.app:app", host="0.0.0.0", port=port, reload=False)
