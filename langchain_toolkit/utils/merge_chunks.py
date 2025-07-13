"""
流式响应的结果处理方法。
"""

from __future__ import annotations

from langchain_core.messages import message_chunk_to_message

from typing import TYPE_CHECKING, cast
if TYPE_CHECKING:
    from langchain_core.messages import BaseMessageChunk, BaseMessage
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AnyMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate


# ====主要方法。====
def merge_chunks_into_message(
    chunks: list[BaseMessageChunk],
) -> BaseMessage:
    """
    将流式传输得到的chunks合并为message。

    实现方法:
        - BaseMessageChunk已经重载了add运算符，即 __add__ 方法，可以直接使用 '+' 进行操作。(BaseMessage其实也可以。)
            进一步，直接使用sum方法简化操作。
        - 使用langchain_core中提供的message_chunk_to_message方法，更完善的完成转换。
    这2个方法均可完全解决content字段以外字段的各种情况。

    Args:
        chunks (list[BaseMessageChunk]): 流式传输获得的chunks。
            - 一般为AIMessageChunk。
            - 前置操作需要接收流式响应的结果并合并为list。

    Returns:
        BaseMessage: 合并的结果。
            没有类型指定，会还原为BaseMessage。可以使用isinstance或者cast方法。
    """
    message = message_chunk_to_message(
        chunk=sum(chunks, chunks[0]),  # 直接使用sum方法，简化合并chunk操作。BaseMessageChunk实现了 '+' 运算符。
    )
    return message


# ====示例代码。====
async def a_call_llm_demo(
    chat_prompt_template: ChatPromptTemplate,
    llm: BaseChatModel,
    chat_history: list[AnyMessage],
) -> AIMessage:
    """
    这仅仅是一段示例代码。通过LLM获取流式响应，并将其处理为正常可进入chat-history的记录。
    """
    llm_chain = chat_prompt_template | llm
    chunks = []
    async for chunk in llm_chain.astream(input={'chat_history': chat_history},):
        chunks.append(chunk)
    response = merge_chunks_into_message(cast('list[AIMessageChunk]', chunks))
    response = cast('AIMessage', response)
    return response

