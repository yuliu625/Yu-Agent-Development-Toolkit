"""
测试会使用到的LLM。
"""

from __future__ import annotations
import pytest

from src.langchain_toolkit import BaseLLMFactory


# if TYPE_CHECKING:


@pytest.fixture
def qwen_plus():
    qwen_plus = BaseLLMFactory().create_dashscope_llm('qwen-plus')
    return qwen_plus

