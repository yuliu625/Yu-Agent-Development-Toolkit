"""
LLM的测试工程。

不用全部测试，仅测试需要使用的LLM。

仅基础模型测试，面临场景:
    - 新环境: API是否可用，当前网络是否可连接。
    - 新模型: 模型参数指定情况，模型响应格式。

更复杂的情况需要构建相关的chat-history进行请求。
"""

from __future__ import annotations
import pytest

from langchain_toolkit.model_factory.base_llm_factory import BaseLLMFactory

from langchain_core.messages import AIMessage

from typing import TYPE_CHECKING
# if TYPE_CHECKING:

_text_message_1 = "你的版本型号是什么？"


class TestBaseLLMs:
    """
    根据具体项目需求设置每个测试函数的参数。
    """
    @pytest.mark.parametrize(
        'model_name, message', [
            ('gpt-4o-mini', _text_message_1),
        ])
    def test_openai_llm(self, model_name, message):
        llm = BaseLLMFactory.get_openai_llm(model_name=model_name)
        response = llm.invoke(message)
        print(response)
        assert isinstance(response, AIMessage)

    @pytest.mark.parametrize(
        'model_name, message', [
            ('gemini-2.5-flash', _text_message_1),
        ])
    def test_google_llm(self, model_name, message):
        llm = BaseLLMFactory.get_google_llm(model_name=model_name)
        response = llm.invoke(message)
        print(response)
        assert isinstance(response, AIMessage)

    @pytest.mark.parametrize(
        'model_name, message', [
            ('claude-opus-4', _text_message_1),
        ])
    def test_anthropic_llm(self, model_name, message):
        llm = BaseLLMFactory.get_anthropic_llm(model_name=model_name)
        response = llm.invoke(message)
        print(response)
        assert isinstance(response, AIMessage)

    @pytest.mark.parametrize(
        'model_name, message', [
            ('qwen-plus', _text_message_1),
        ])
    def test_dashscope_llm(self, model_name, message):
        llm = BaseLLMFactory.get_dashscope_llm(model_name=model_name)
        response = llm.invoke(message)
        print(response)
        assert isinstance(response, AIMessage)

    @pytest.mark.parametrize(
        'model_name, message', [
            ('deepseek-chat', _text_message_1),
        ])
    def test_deepseek_llm(self, model_name, message):
        llm = BaseLLMFactory.get_deepseek_llm(model_name=model_name)
        response = llm.invoke(message)
        print(response)
        assert isinstance(response, AIMessage)

