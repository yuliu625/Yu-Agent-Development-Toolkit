"""
对于LRM的响应的处理方法。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class ReasoningContentProcessor:
    @staticmethod
    def extract_reasoning_content_from_addition_kwargs(
        ai_message: AIMessage,
    ) -> str:
        return ai_message.additional_kwargs.get('reasoning_content', "")

    @staticmethod
    def extract_thinking_content_from_message_content(
        ai_message: AIMessage,
    ) -> str:
        ...

    @staticmethod
    def thinking_to_reasoning(
        ai_message: AIMessage,
    ) -> AIMessage:
        ...

    @staticmethod
    def reasoning_to_thinking(
        ai_message: AIMessage,
    ) -> AIMessage:
        ...

    @staticmethod
    def delete_thinking_content(
        ai_message: AIMessage,
    ) -> AIMessage:
        ...

