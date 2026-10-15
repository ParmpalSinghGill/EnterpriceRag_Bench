"""Pure LlamaIndex QueryEngine and ReActAgent assembly."""

from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.tools import QueryEngineTool

from pure_llamaindex.index import get_vector_index


def get_query_engine(similarity_top_k: int = 7) -> BaseQueryEngine:
    """Build pure LlamaIndex vector query engine."""
    index = get_vector_index()
    return index.as_query_engine(
        similarity_top_k=similarity_top_k,
        llm=Settings.llm,
    )


def get_react_agent(similarity_top_k: int = 7) -> ReActAgent:
    """Build pure LlamaIndex ReAct Agent with vector search tool."""
    query_engine = get_query_engine(similarity_top_k=similarity_top_k)

    tool = QueryEngineTool.from_defaults(
        query_engine,
        name="enterprise_knowledge_search",
        description=(
            "Search internal engineering documents, incident postmortems, "
            "technical designs, and ADRs."
        ),
    )

    return ReActAgent(
        name="EnterpriseAgent",
        description="Autonomous assistant answering engineering and architecture questions.",
        tools=[tool],
        llm=Settings.llm,
        verbose=True,
    )
