"""LlamaIndex QueryEngine and ReActAgent Factory.

Exposes clean, one-line initializers for:
1. get_query_engine(): Deterministic RAG query engine
2. get_agent(): Autonomous ReAct agent with enterprise search tools
"""

from __future__ import annotations

from typing import Optional

from llama_index.core import get_response_synthesizer
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from llamaindex_app.llm import NemotronLLM
from llamaindex_app.retriever import EnterpriseHybridRetriever

DEFAULT_QA_PROMPT = PromptTemplate(
    "Context:\n{context_str}\n\nQuestion: {query_str}\nAnswer: "
)


def get_query_engine(
    top_k: int = 7,
    use_expansion: bool = True,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> RetrieverQueryEngine:
    """Build a deterministic LlamaIndex RAG QueryEngine with Top-7 hybrid search."""
    retriever = EnterpriseHybridRetriever(top_k=top_k, use_expansion=use_expansion)
    llm = NemotronLLM(model=model, temperature=temperature)
    synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode="compact",
        text_qa_template=DEFAULT_QA_PROMPT,
    )
    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
    )


def get_agent(
    top_k: int = 7,
    use_expansion: bool = True,
    model: Optional[str] = None,
    verbose: bool = False,
) -> ReActAgent:
    """Build an autonomous LlamaIndex ReAct Agent with enterprise search capabilities."""
    query_engine = get_query_engine(top_k=top_k, use_expansion=use_expansion, model=model)
    llm = NemotronLLM(model=model, temperature=0.0)

    tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="enterprise_search",
            description=(
                "Search internal company documents across Slack, Jira, Confluence, Linear, "
                "GitHub, Google Drive, Gmail, HubSpot, and Fireflies transcripts."
            ),
        ),
    )

    system_prompt = (
        "You are an expert enterprise AI assistant for Redwood Inference. "
        "When asked questions about company projects, incidents, configurations, or customers, "
        "use the 'enterprise_search' tool to look up facts. "
        "Always provide accurate, specific facts (numbers, dates, IDs). "
        "If the information is not found, state clearly that it was not found."
    )

    return ReActAgent(
        name="EnterpriseRAGAgent",
        description="Autonomous agent for querying internal company documents",
        system_prompt=system_prompt,
        tools=[tool],
        llm=llm,
        verbose=verbose,
    )

