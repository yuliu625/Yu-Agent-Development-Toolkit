"""

"""

from langgraph.graph import StateGraph

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # import MASState here
    from langgraph.graph.state import CompiledStateGraph


class BaseGraphBuilder:
    def __init__(self, state):
        self.graph_builder = StateGraph(state)

    def build_graph(
        self,
    ) -> CompiledStateGraph:
        self._add_nodes()
        self._add_edges()
        graph = self.graph_builder.compiled()
        return graph

    def _add_nodes(self):
        """
        注册MAS的nodes。
        """

    def _add_edges(self):
        """
        注册MAS的edges。
        """

