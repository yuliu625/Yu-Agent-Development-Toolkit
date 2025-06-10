"""
LLM的测试工程。

不用全部测试，仅测试需要使用的LLM。
"""

from __future__ import annotations
import pytest

from langchain_toolkit.model_factory.base_llm_factory import BaseLLMFactory
from langchain_core.messages import AIMessage

from typing import TYPE_CHECKING
# if TYPE_CHECKING:

_text_message_1 = "你的版本型号是什么？"


class TestLLMs:
    @pytest.mark.parametrize(
        'model, message', [
            ('gpt-4o-mini', _text_message_1),
        ])
    def test_openai_llm(self, model, message):
        llm = BaseLLMFactory.get_openai_llm(model_name=model)
        response = llm.invoke(message)
        print(response)
        assert isinstance(response, AIMessage)

    @pytest.mark.parametrize(
        'model, message', [
            ('gemini-2.5-flash', _text_message_1),
        ])
    def test_google_llm(self, model, message):
        llm = BaseLLMFactory.get_google_llm(model=model)
        response = llm.invoke(message)
        print(response)
        assert isinstance(response, AIMessage)

    @pytest.mark.parametrize(
        'model, message', [
            ('claude-opus-4', _text_message_1),
        ])
    def test_anthropic_llm(self, model, message):
        llm = BaseLLMFactory.get_anthropic_llm(model_name=model)
        response = llm.invoke(message)
        print(response)
        assert isinstance(response, AIMessage)

    @pytest.mark.parametrize(
        'model, message', [
            ('qwen-plus', _text_message_1),
        ])
    def test_dashscope_llm(self, model, message):
        llm = BaseLLMFactory.get_dashscope_llm(model_name=model)
        response = llm.invoke(message)
        print(response)
        assert isinstance(response, AIMessage)

    @pytest.mark.parametrize(
        'model, message', [
            ('deepseek-chat', _text_message_1),
        ])
    def test_deepseek_llm(self, model, message):
        llm = BaseLLMFactory.get_deepseek_llm(model)
        response = llm.invoke(message)
        print(response)
        assert isinstance(response, AIMessage)

