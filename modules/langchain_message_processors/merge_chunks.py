"""
Sources:
    https://github.com/yuliu625/Yu-Agent-Development-Toolkit/blob/main/modules/content_processors/langchain_message_processors/merge_chunks.py

References:
    langchain_core.messages::message_chunk_to_message

Synopsis:
    流式响应的结果处理方法。

Notes:
    少数场景下，需要对于响应的 chunks 进行合并处理。

    自己写的方法总是不够好，例如 metadata 的处理。
    我在 langchain_core 的源码中发现了这个预置方法。

    更好的做法:
        在独立的 gateway 中进行处理。如果有需要，自构建 hook 。
        优先 stream -> message 。
        所有的 message 处理应该独立于 agent 。
"""

from __future__ import annotations
from loguru import logger

from langchain_core.messages import message_chunk_to_message

from typing import TYPE_CHECKING, cast
if TYPE_CHECKING:
    from langchain_core.messages import BaseMessageChunk, BaseMessage
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AnyMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate


# ==== 主要方法。 ====
def merge_chunks_into_message(
    chunks: list[BaseMessageChunk],
) -> BaseMessage:
    """
    将流式传输得到的 chunks 合并为 message 。

    实现方法:
        - BaseMessageChunk 已经重载了 add 运算符，即 __add__ 方法，可以直接使用 '+' 进行操作。( BaseMessage 其实也可以。)
            进一步，直接使用 sum 方法简化操作。
        - 使用 langchain_core 中提供的 message_chunk_to_message 方法，更完善的完成转换。
    这2个方法均可完全解决 content 字段以外字段的各种情况。

    Args:
        chunks (list[BaseMessageChunk]): 流式传输获得的 chunks 。
            - 一般为 AIMessageChunk 。
            - 前置操作需要接收流式响应的结果并合并为 list 。

    Returns:
        BaseMessage: 合并的结果。
            没有类型指定，会还原为 BaseMessage 。可以使用 isinstance 或者 cast 方法。
    """
    message = message_chunk_to_message(
        chunk=sum(chunks, chunks[0]),  # 直接使用 sum 方法，简化合并 chunk 操作。BaseMessageChunk 实现了 '+' 运算符。
    )
    return message


# ==== demo usage code ====
async def a_call_llm_demo(
    chat_prompt_template: ChatPromptTemplate,
    llm: BaseChatModel,
    chat_history: list[AnyMessage],
) -> AIMessage:
    """
    这仅仅是一段示例代码。通过 LLM 获取流式响应，并将其处理为正常可进入 chat-history 的记录。
    """
    llm_chain = chat_prompt_template | llm
    chunks = []
    async for chunk in llm_chain.astream(input={'chat_history': chat_history},):
        chunks.append(chunk)
    response = merge_chunks_into_message(cast('list[AIMessageChunk]', chunks))
    response = cast('AIMessage', response)
    return response

