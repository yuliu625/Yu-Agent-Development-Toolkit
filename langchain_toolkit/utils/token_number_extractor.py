"""
Sources:
    https://github.com/yuliu625/Yu-Agent-Development-Toolkit/

References:
    None

Synopsis:
    从 AIMessage 中获得消耗的 token 数量。

Notes:
    这个工具根据 langchain 中具体厂商的返回信息格式构建，会持续更新。

    Refactor:
        在通过 gateway 统一构建和管理后，可仅保留 openai 一种方法。
"""

from __future__ import annotations
from loguru import logger

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from langchain_core.messages import AIMessage


class TokenNumberExtractor:
    """
    根据具体厂商返回的 AIMessage 的 schema ，针对性提取代表当前 ai_message 的 token 数量。
    """

    @staticmethod
    def extract_token_number_from_ai_message(
        ai_message: AIMessage,
        model_client: Literal['openai', 'google', 'dashscope', 'deepseek'],
    ) -> int:
        if model_client == 'openai':
            return TokenNumberExtractor.extract_openai_ai_message_token_number(ai_message=ai_message)
        elif model_client == 'google':
            return TokenNumberExtractor.extract_google_ai_message_token_number(ai_message=ai_message)
        elif model_client == 'dashscope':
            return TokenNumberExtractor.extract_dashscope_ai_message_token_number(ai_message=ai_message)
        elif model_client == 'deepseek':
            return TokenNumberExtractor.extract_deepseek_ai_message_token_number(ai_message=ai_message)

    @staticmethod
    def extract_openai_ai_message_token_number(
        ai_message: AIMessage,
    ) -> int:
        token_number = ai_message.response_metadata['token_usage']['completion_tokens']
        assert isinstance(token_number, int)
        return token_number

    @staticmethod
    def extract_google_ai_message_token_number(
        ai_message: AIMessage,
    ) -> int:
        token_number = ai_message.response_metadata['token_usage']['completion_tokens']
        assert isinstance(token_number, int)
        return token_number

    @staticmethod
    def extract_dashscope_ai_message_token_number(
        ai_message: AIMessage,
    ) -> int:
        """
        仅 dashscope 的返回信息和其他 client 不一样。
        """
        token_number = ai_message.response_metadata['token_usage']['output_tokens']
        assert isinstance(token_number, int)
        return token_number

    @staticmethod
    def extract_deepseek_ai_message_token_number(
        ai_message: AIMessage,
    ) -> int:
        token_number = ai_message.response_metadata['token_usage']['completion_tokens']
        assert isinstance(token_number, int)
        return token_number

