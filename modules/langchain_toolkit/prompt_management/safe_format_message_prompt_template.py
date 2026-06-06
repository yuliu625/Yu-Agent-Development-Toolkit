"""
Sources:
    https://github.com/yuliu625/Yu-Agent-Development-Toolkit/blob/main/modules/langchain_toolkit/prompt_management/safe_format_message_prompt_template.py

References:
    None

Synopsis:
    对 message-prompt-template 的安全 format 方法。

Notes:
    约定行为:
        - 基础的 chat-prompt-template 由 system-message 和 messages-place-holder 构成。
        - chat-history 与独立管理，拥有独立逻辑。

    为什么不使用 ChatPromptTemplate 的 add 方法。
        - 不和 system-prompt 独立。
        - 如果进行过 partial 操作， langchain v0.3 并不会其进行处理。
"""

from __future__ import annotations
from loguru import logger

from langchain_core.prompts import ChatPromptTemplate

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.prompts.message import BaseMessagePromptTemplate


def safe_format_message_prompt_template(
    message_prompt_templates: list[BaseMessagePromptTemplate | BaseMessage],
    format_kwargs: dict,
) -> list[BaseMessage]:
    """
    对 message-prompt-template 的安全 format 方法。

    避免直接 format 造成变量不匹配但错误不识别。

    实现:
        - 使用 ChatPromptTemplate 加载原本的 MessagePromptTemplate 和 BaseMessage ，进行变量识别，可以使用 invoke 方法。
        - 使用 invoke 方法，如果发生变量不匹配会报错。
        - 返回处理好的 list 。

    Args:
        message_prompt_templates (list[BaseMessagePromptTemplate | BaseMessage]): 需要进行 format 的 list 。
        format_kwargs (dict): 指定的 format 的映射。如果不指定需要映射为 None 。

    Returns:
        list[BaseMessage]: 处理好的 list 。
    """
    chat_prompt_template = ChatPromptTemplate(
        messages=message_prompt_templates,
    )
    chat_prompt_value = chat_prompt_template.invoke(input=format_kwargs)
    messages = chat_prompt_value.to_messages()
    # assert all(isinstance(message, BaseMessage) for message in messages)
    return messages

