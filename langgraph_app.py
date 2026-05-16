"""
LangGraph Studio entry-point.

This module exposes the compiled SHL assessment graph so that
`langgraph dev` (LangGraph Studio) can discover and visualize it.

The `langgraph.json` config points here:
    "shl_agent": "langgraph_app:graph"
"""
from src.core.graph import SHLGraphBuilder

# Build the graph and expose the compiled form at module level.
# LangGraph Studio requires a module-level compiled graph object.
_builder = SHLGraphBuilder()

# `graph` is the compiled StateGraph — this is what langgraph.json references.
graph = _builder.graph
