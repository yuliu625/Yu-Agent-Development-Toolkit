"""
对message-prompt-template的安全format方法。

约定行为:
    - 基础的chat-prompt-template由system-message和messages-place-holder构成。
    - chat-history与独立管理，拥有独立逻辑。

为什么不使用ChatPromptTemplate的add方法。
    - 不和system-prompt独立。
    - 如果进行过partial操作，langchain0.3并不会其进行处理。
"""

from __future__ import annotations

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
    对message-prompt-template的安全format方法。

    避免直接format造成变量不匹配但错误不识别。

    实现:
        - 使用ChatPromptTemplate加载原本的MessagePromptTemplate和BaseMessage，进行变量识别，可以使用invoke方法。
        - 使用invoke方法，如果发生变量不匹配会报错。
        - 返回处理好的list。

    Args:
        message_prompt_templates (list[BaseMessagePromptTemplate | BaseMessage]): 需要进行format的list。
        format_kwargs (dict): 指定的format的映射。如果不指定需要映射为None。

    Returns:
        list[BaseMessage]: 处理好的list。
    """
    chat_prompt_template = ChatPromptTemplate(
        messages=message_prompt_templates,
    )
    chat_prompt_value = chat_prompt_template.invoke(input=format_kwargs)
    messages = chat_prompt_value.to_messages()
    # assert all(isinstance(message, BaseMessage) for message in messages)
    return messages

