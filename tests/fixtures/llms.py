"""
测试会使用到的LLM。
"""

from __future__ import annotations
import pytest

from langchain_toolkit.model_factory.llm_factory import LLMFactory

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


@pytest.fixture
def qwen_plus():
    qwen_plus = LLMFactory().get_dashscope_llm('qwen-plus')
    return qwen_plus

