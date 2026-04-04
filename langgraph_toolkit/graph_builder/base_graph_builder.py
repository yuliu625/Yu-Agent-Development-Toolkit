"""
Sources:
    https://github.com/yuliu625/Yu-Agent-Development-Toolkit/langgraph_toolkit/graph_builder/base_graph_builder.py

References:
    None

Synopsis:
    graph builder boilerplate

Notes:
    复制下面的代码，然后在具体项目中进行修改。

    Refactor:
        以下构造器是有状态的，可以函数式重构，以实现更加显式和可信。
"""

from __future__ import annotations
from loguru import logger

# from ? import MASState  # 由于构建 graph_builder 需要使用到 MASState ，因此不能仅以类型声明。

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
        state#: type[MASState],
    ):
        # 初始化计算图构建。实际中不会以变量传入 state ，因为 state 为数据类，更多实现方法为以包导入并硬编码。
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
        注册 MAS 的 nodes 。
        """

    def _add_edges(self):
        """
        注册 MAS 的 edges 。
        """

