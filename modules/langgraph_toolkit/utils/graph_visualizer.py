"""
Sources:
    https://github.com/yuliu625/Yu-Agent-Development-Toolkit/blob/main/modules/langgraph_toolkit/utils/graph_visualizer.py

References:
    None

Synopsis:
    可视化查看构建的 graph 的方法。

Notes:

"""

from __future__ import annotations
from loguru import logger

from IPython.display import Image, display

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


class GraphVisualizer:
    """
    可视化查看 graph 的方法。

    封装了 mermaid 相关获取可视化 graph 的方法，主要方法是打印 mermaid-code 。
    """

    # ====主要方法。====
    @staticmethod
    def get_mermaid_code(
        graph: CompiledStateGraph,
    ) -> str:
        """
        最主要的方法。获取 mermaid-code 。

        在 https://mermaid.live/ 中复制输出的 code ，然后查看图像。

        Args:
            graph (CompiledStateGraph): 已经编译好的 graph 。

        Returns:
            str: mermaid-code，使用 print 打印返回结果，然后复制。
        """
        mermaid_code = graph.get_graph(xray=True).draw_mermaid()
        # print(mermaid_code)
        return mermaid_code

    # ====备用方法。====
    @staticmethod
    def get_mermaid_png(
        graph: CompiledStateGraph,
    ) -> Image:
        """
        获取 mermaid 的图片。

        默认使用 MermaidDrawMethod.API ，可能会遇到网络问题。
        可以使用其他 MermaidDrawMethod ，但是需要额外依赖，并且都不好用。

        可进行修改，增加图片渲染相关的 kwargs 。

        Args:
            graph (CompiledStateGraph): 已经编译好的 graph 。

        Returns:
            Image: png 图片，默认渲染效果不好。
                在 ipynb 中，可以直接打印 graph 生成图片，不需要该方法。
        """
        mermaid_png = Image(graph.get_graph(xray=True).draw_mermaid_png())
        display(mermaid_png)
        return mermaid_png

    # ====备用方法。====
    @staticmethod
    def save_mermaid_png(
        graph: CompiledStateGraph,
        output_file_path: str,
    ) -> Image:
        """
        保存 mermaid 的图片。

        Args:
            graph (CompiledStateGraph): 已经编译好的 graph 。
            output_file_path (str): 保存图片的路径。

        Returns:
            Image: png 图片，渲染的时候保存至指定路径。
        """
        mermaid_png = Image(graph.get_graph(xray=True).draw_mermaid_png(output_file_path=output_file_path))
        return mermaid_png

