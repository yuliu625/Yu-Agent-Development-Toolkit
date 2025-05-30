"""
可视化查看构建的graph的方法。
"""

from __future__ import annotations

from IPython.display import Image, display

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


class GraphVisualizer:
    """
    可视化查看graph的方法。

    封装了mermaid相关获取可视化graph的方法，主要方法是打印mermaid-code。
    """
    @staticmethod
    def get_mermaid_code(graph: CompiledStateGraph) -> str:
        """
        最主要的方法。获取mermaid-code。

        在 https://mermaid.live/ 中复制输出的code，然后查看图像。

        Args:
            graph: 已经编译好的graph。

        Returns:
            (str), mermaid-code，使用print打印返回结果，然后复制。
        """
        mermaid_code = graph.get_graph().draw_mermaid()
        # print(mermaid_code)
        return mermaid_code

    @staticmethod
    def get_mermaid_png(graph: CompiledStateGraph) -> Image:
        """
        获取mermaid的图片。

        默认使用MermaidDrawMethod.API，可能会遇到网络问题。
        可以使用其他MermaidDrawMethod，但是需要额外依赖，并且都不好用。

        可进行修改，增加图片渲染相关的kwargs。

        Args:
            graph: 已经编译好的graph。

        Returns:
            (Image), png图片，默认渲染效果不好。
            在ipynb中，可以直接打印graph生成图片，不需要该方法。。
        """
        mermaid_png = Image(graph.get_graph().draw_mermaid_png())
        display(mermaid_png)
        return mermaid_png

    @staticmethod
    def save_mermaid_png(
        graph: CompiledStateGraph,
        output_file_path: str,
    ) -> Image:
        """
        保存mermaid的图片。

        Args:
            graph: 已经编译好的graph。
            output_file_path: 保存图片的路径。

        Returns:
            (Image), png图片，渲染的时候保存至指定路径。
        """
        mermaid_png = Image(graph.get_graph().draw_mermaid_png(output_file_path=output_file_path))
        return mermaid_png

