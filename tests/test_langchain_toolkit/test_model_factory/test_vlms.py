"""
VLM的测试工程。
"""

from __future__ import annotations
import pytest

from langchain_toolkit.model_factory.vlm_factory import VLMFactory

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class TestBaseVLMs:
    """
    暂未实现的主要原因:
        - 在llama-index，一些单模态和多模态的模型是有区别的。
            但是，在langchain中，单模态和多模态通常统一被chat-model支持。
    """

