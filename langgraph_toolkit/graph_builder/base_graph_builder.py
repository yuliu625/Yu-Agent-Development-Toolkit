"""
这是一个graph-builder的Boilerplate。复制下面的代码，然后在具体项目中进行修改。
"""

from __future__ import annotations

# import MASState here  # 由于构建graph_builder需要使用到MASState，因此不能仅以类型声明。

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.checkpoint.base import BaseCheckpointSaver


class BaseGraphBuilder:
    """
    计算图的构造器。
    """
    def __init__(
        self,
        state,
    ):
        # 初始化计算图构建。实际中不会以变量传入state，因为state为数据类，更多实现方法为以包导入并写死。
        self.graph_builder = StateGraph(state)
        # 注册需要的工具。

    def build_graph(
        self,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> CompiledStateGraph:
        self._add_nodes()
        self._add_edges()
        graph = self.graph_builder.compile(checkpointer=checkpointer)
        return graph

    def _add_nodes(self):
        """
        注册MAS的nodes。
        """

    def _add_edges(self):
        """
        注册MAS的edges。
        """

