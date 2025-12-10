"""
辅助进行结构化输出工具。

让LLM直接在分析完成后进行结构化输出可能的问题:
    - 有时会影响LLM的能力
    - 以重试机制实现，在长上下文成本可能昂贵。

让另一个LLM总结并进行结构化输出:
    - 异构agent，更高的灵活性。
    - 仅一条文本，价格低，任务简单。
    - 但是，提取的结果可能不是原始agent的本意。
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from typing import TYPE_CHECKING, cast
if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from pydantic import BaseModel


class StructuredOutputHelper:
    """
    从文本中提取结构化数据的方法。
    """
    def __init__(
        self,
        llm: BaseChatModel,
        schema_pydantic_base_model: type[BaseModel],
        system_message: SystemMessage,
        max_retries: int = 3,
    ):
        """
        构造结构化输出提取工具的必要参数。

        Args:
            llm (BaseChatModel): 基础的用于推理的基座模型。需要具有结构化提取功能。
            schema_pydantic_base_model (BaseModel): 基于pydantic定义的schema。
            system_message (SystemMessage): 提取指令。在这个实现中，它不与llm绑定，可进行修改。
            max_retries (int): 最大重试次数。基于runnable本身的实现。
        """
        self._system_message = system_message

        self._structured_llm = self._build_structured_llm(
            llm=llm,
            schema_pydantic_base_model=schema_pydantic_base_model,
            max_retries=max_retries,
        )

    # ====暴露方法。====
    async def extract_structured_data(
        self,
        raw_str: str,
    ) -> BaseModel:
        """
        提取结构化数据的方法。

        这个方法默认:
            - 文本数据。目前仅对于文本进行提取。
                - 如果需要多模态数据，可以多一步转换，结果更加可靠和稳定。

        Args:
            raw_str (str): 原始被提取的文本。

        States:
            _structured_llm (BaseChatModel): 构造好的可以进行结构化提取的llm。

        Returns:
            BaseModel: 基于pydantic定义的schema的数据对象。
                调用该方法的函数，可以进一步确认提取的schema的定义。
        """
        response = await self._structured_llm.ainvoke(
            input=[
                self._system_message,
                HumanMessage(content=raw_str),
            ]
        )
        return response

    # ====内部构建方法。====
    def _build_structured_llm(
        self,
        llm: BaseChatModel,
        schema_pydantic_base_model: type[BaseModel],
        # system_message: SystemMessage,
        max_retries: int = 3,
    ) -> BaseChatModel:
        """
        构造structured_llm的方法。

        这个实现默认:
            - 不对于system-message进行限制，不将system-message与llm提前绑定。
                - 但相关控制需要使用structured_llm的方法实现。

        Args:
            llm (BaseChatModel): 基础的用于推理的基座模型。需要具有结构化提取功能。
            schema_pydantic_base_model (BaseModel): 基于pydantic定义的schema。
            max_retries (int): 最大重试次数。基于runnable本身的实现。

        Returns:
            BaseChatModel: 被限制为仅会进行结构化输出的structured_llm。
        """
        structured_llm = llm.with_structured_output(
            schema=schema_pydantic_base_model,
        ).with_retry(
            stop_after_attempt=max_retries,
        )
        structured_llm = cast('BaseChatModel', structured_llm)
        return structured_llm

