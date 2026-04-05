"""
Sources:
    https://github.com/yuliu625/Yu-Agent-Development-Toolkit/blob/main/agnostic_utils/content_annotator.py

References:
    None

Synopsis:
    对文档内容进行标注的方法。

Notes:
    对文本内容进行标注的方法。

    该实现:
        - 无依赖: 仅是字符串处理方法。
        - 标注格式:
            - html comment
            - xml
"""

from __future__ import annotations
from loguru import logger

import html
import re

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class ContentAnnotator:
    """
    内容注释器。
    """
    @staticmethod
    def safe_annotate_with_html(
        tag: str,
        original_text: str,
    ) -> str:
        # 执行检测。
        stripped_text = original_text.strip()
        if stripped_text.startswith(f"<!--{tag}-start-->") and stripped_text.endswith(f"<!--{tag}-end-->"):
            # model 自己加了 tag 。
            logger.info(f"HTML annotation {tag} already exists.")
            return original_text
        else:
            # 正常添加 tag 。
            return ContentAnnotator.annotate_with_html(
                tag=tag,
                original_text=original_text,
            )

    @staticmethod
    def safe_annotate_with_xml(
        tag: str,
        original_text: str,
    ) -> str:
        # 执行检测。
        stripped_text = original_text.strip()
        if stripped_text.startswith(f"<{tag}>") and stripped_text.endswith(f"</{tag}>"):
            # model 自己加了 tag 。
            logger.info(f"XML annotation {tag} already exists.")
            return original_text
        else:
            # 正常添加 tag 。
            return ContentAnnotator.annotate_with_xml(
                tag=tag,
                original_text=original_text,
            )

    @staticmethod
    def annotate_with_html(
        tag: str,
        original_text: str,
    ) -> str:
        """
        给一段字符串以 html 注释的方式添加标注。

        可以使用的场景:
            - MAS 中，一个 agent 会与多个 agent 交互。以此区别 HumanMessage 的实际身份。
            - RAG 中，区分文档和查询。

        Args:
            tag (str): Agent的名称。
            original_text (str): 原始字符串。

        Returns:
            str: 包裹了 html 注释的字符串。
        """
        result = (
            f"<!--{tag}-start-->\n"
            + original_text
            + f"\n<!--{tag}-end-->"
        )
        logger.trace(f"Annotation with html tag: {tag}")
        logger.trace(f"Annotation result: {result}")
        return result

    @staticmethod
    def annotate_with_xml(
        tag: str,
        original_text: str,
    ) -> str:
        result = (
            f"<{tag}>\n"
            + original_text
            + f"\n</{tag}>"
        )
        logger.trace(f"Annotation with xml tag: {tag}")
        logger.trace(f"Annotation result: {result}")
        return result

