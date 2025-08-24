"""
从AIMessage中获得消耗的token数量。

这个工具根据langchain中具体厂商的返回信息格式构建，会持续更新。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from langchain_core.messages import AIMessage


class TokenNumberExtractor:
    """
    根据具体厂商返回的AIMessage的schema，针对性提取代表当前ai_message的token数量。
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
        仅dashscope的返回信息和其他client不一样。
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

