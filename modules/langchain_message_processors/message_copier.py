"""
Sources:
    https://github.com/yuliu625/Yu-Agent-Development-Toolkit/blob/main/modules/content_processors/langchain_message_processors/message_copier.py

References:
    langchain_core.messages::BaseMessage
    langchain_core.messages::AIMessage
    langchain_core.messages::HumanMessage

Synopsis:
    复制 langchain 中各种 BaseMessage 派生对象的方法。

Notes:
    使用场景:
        - 需要修改 BaseMessage ，但是为了满足函数式编程，不出现副作用。

    原理:
        - langchain 中的 Serializable 是基于 pydantic 的 BaseModel 的，可以直接使用相关方法例如 .model_dump 方法。

    其他可以考虑的方法:
        - langchain_core.messages.base::message_to_dict
"""

from __future__ import annotations
from loguru import logger

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


class MessageCopier:
    """
    复制 langchain 中各种 BaseMessage 派生对象的方法。
    """
    @staticmethod
    def auto_copy_message(
        original_message: BaseMessage,
    ) -> AIMessage | HumanMessage:
        """
        复制一条 BaseMessage，自动推断类型，并复制为相同类型的。

        Args:
            original_message (BaseMessage): 原始需要被复制的 message 。

        Returns:
            Union[AIMessage, HumanMessage]: 复制的 BaseMessage ，修改与原始 message 独立。
        """
        message_dict = original_message.model_dump()
        if isinstance(original_message, AIMessage):
            return AIMessage(**message_dict)
        elif isinstance(original_message, HumanMessage):
            return HumanMessage(**message_dict)
        else:
            raise TypeError

    @staticmethod
    def copy_message(
        original_message: BaseMessage,
        output_message_type: Literal['ai_message', 'human_message'],
    ) -> AIMessage | HumanMessage:
        """
        复制一条 BaseMessage ，需要指定复制后的类型，可以是不同类型。

        Args:
            original_message (BaseMessage): 原始需要被复制的 message 。
            output_message_type (Literal['ai_message', 'human_message']): 复制的 message 的具体类型。

        Returns:
            Union[AIMessage, HumanMessage]: 复制的 BaseMessage ，修改与原始 message 独立。
        """
        message_dict = original_message.model_dump()
        if output_message_type == 'ai_message':
            return AIMessage(**message_dict)
        elif output_message_type == 'human_message':
            return HumanMessage(**message_dict)
        else:
            raise TypeError

