from langgraph.graph import (
    StateGraph,
    START,
    END
)
import traceback
from langchain_core.messages import HumanMessage, AIMessage

_ALLOWED_ROLES = {"user", "assistant", "human", "ai", "function", "tool", "system", "developer"}


def _normalize_messages(messages):
    """Normalize message roles so LangGraph's add_messages reducer never sees unknown types."""
    normalized = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role not in _ALLOWED_ROLES:
                role = "user"  # treat Swagger default 'string' or any unknown as user
            normalized.append({"role": role, "content": content})
        else:
            normalized.append(msg)
    return normalized

from src.prompts.schemas import GraphState

from src.core.analyzer import (

    analyzer_node,

    refusal_node,

    clarification_node,

    comparison_node,

    completion_node,

    recommendation_node,

    route_node
)


class SHLGraphBuilder:


    def __init__(self):

        self.graph_builder = StateGraph(GraphState)

        self._add_nodes()

        self._add_edges()

        self.graph = (
            self.graph_builder.compile()
        )


    def _add_nodes(self):

        self.graph_builder.add_node(
            "analyzer",
            analyzer_node
        )

        self.graph_builder.add_node(
            "refusal_node",
            refusal_node
        )

        self.graph_builder.add_node(
            "clarification_node",
            clarification_node
        )

        self.graph_builder.add_node(
            "comparison_node",
            comparison_node
        )

        self.graph_builder.add_node(
            "completion_node",
            completion_node
        )

        self.graph_builder.add_node(
            "recommendation_node",
            recommendation_node
        )

    def _add_edges(self):

        self.graph_builder.add_edge(
            START,
            "analyzer"
        )
        self.graph_builder.add_conditional_edges(
            "analyzer",
            route_node,
            {
                "refusal_node":
                    "refusal_node",
                "clarification_node":
                    "clarification_node",
                "comparison_node":
                    "comparison_node",
                "completion_node":
                    "completion_node",
                "recommendation_node":
                    "recommendation_node"
            }
        )
        self.graph_builder.add_edge(
            "refusal_node",
            END
        )
        self.graph_builder.add_edge(
            "clarification_node",
            END
        )
        self.graph_builder.add_edge(
            "comparison_node",
            END
        )
        self.graph_builder.add_edge(
            "completion_node",
            END
        )
        self.graph_builder.add_edge(
            "recommendation_node",
            END
        )

    def invoke(self, messages):

        try:

            # Normalize roles before LangGraph's add_messages reducer sees them
            messages = _normalize_messages(messages)

            initial_state = {
                "messages": messages
            }

            result=self.graph.invoke(
                initial_state
            )

            return {
                "reply":
                    result.get(
                        "reply",
                        ""
                    ),

                "recommendations":
                    result.get(
                        "recommendations",
                        []
                    ),

                "end_of_conversation":
                    result.get(
                        "end_of_conversation",
                        False
                    )
            }

        except Exception as e:

            print("\n========== GRAPH ERROR ==========\n")

            traceback.print_exc()

            print("\n=================================\n")

            return {

                "reply":
                    f"Unable to process the request currently. {e}",
    
                "recommendations":
                    [],

                "end_of_conversation":
                    False
            }