"""
对文档内容进行标注的方法。
"""

from __future__ import annotations

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class ContentAnnotator:
    @staticmethod
    def annotate_with_html_comment(
        tag: str,
        original_text: str,
    ) -> str:
        """
        给一段字符串以html注释的方式添加标注。

        可以使用的场景:
            - MAS中，一个agent会与多个agent交互。以此区别HumanMessage的实际身份。
            - RAG中，区分文档和查询。

        Args:
            tag (str): Agent的名称。
            original_text (str): 原始字符串。

        Returns:
            str: 包裹了html注释的字符串。
        """
        return (
            f"<!--{tag}-start-->\n\n"
            + original_text
            + f"\n\n<!--{tag}-end-->"
        )

    @staticmethod
    def annotate_with_xml(
        tag: str,
        original_text: str,
    ) -> str:
        return (
            f"<{tag}>\n\n"
            + original_text
            + f"\n\n</{tag}>"
        )

