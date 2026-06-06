"""
Sources:
    https://github.com/yuliu625/Yu-Agent-Development-Toolkit/blob/main/modules/langchain_toolkit/agents/structured_output_helper.py

References:
    None

Synopsis:
    辅助进行结构化输出工具。

Notes:
    让 LLM 直接在分析完成后进行结构化输出可能的问题:
        - 有时会影响 LLM 的能力
        - 以重试机制实现，在长上下文成本可能昂贵。

    让另一个 LLM 进行结构化输出:
        - 异构 agent，更高的灵活性。
        - 仅一条文本，价格低，任务简单。
        - 但是，提取的结果可能不是原始 agent 的本意。
"""

from __future__ import annotations
from loguru import logger

from langchain_core.messages import SystemMessage, HumanMessage

from typing import TYPE_CHECKING, cast
if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
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
            schema_pydantic_base_model (BaseModel): 基于 pydantic 定义的 schema 。
            system_message (SystemMessage): 提取指令。在这个实现中，它不与 llm 绑定，可进行修改。
            max_retries (int): 最大重试次数。基于 runnable 本身的实现。
        """
        self._system_message = system_message
        self._structured_llm = self._build_structured_llm(
            llm=llm,
            schema_pydantic_base_model=schema_pydantic_base_model,
            max_retries=max_retries,
        )

    # ==== 常见的默认统一方法。 ====
    async def process_state(
        self,
        state,
        config: RunnableConfig,
    ) -> dict:
        """
        一般 MAS 中，所有 agent 的统一的注册方法。

        构建规范:
            - 异步分离: 在这层隔离异步操作。如无必要，仅提供同步版本。
            - 操作分离: 直接获取需要更新的状态，不在这里构建过多逻辑。

        Args:
            state (MASState): Graph 中定义的 state 。之后添加类型标注。
            config (RunnableConfig): runnable 设计的 config 配置。可以不使用，但在复杂图中，可以提供更好的控制。

        Returns:
            dict: 表示更新字段的dict。
        """
        raise NotImplementedError("一般 MAS 中，所有 agent 的统一的注册方法。")

    # ==== 暴露方法。 ====
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
            _structured_llm (BaseChatModel): 构造好的可以进行结构化提取的 LLM 。

        Returns:
            BaseModel: 基于 pydantic 定义的 schema 的数据对象。
                调用该方法的函数，可以进一步确认提取的 schema 的定义。
        """
        response = await self._structured_llm.ainvoke(
            input=[
                self._system_message,
                HumanMessage(content=raw_str),
            ]
        )
        return response

    # ==== 内部构建方法。 ====
    def _build_structured_llm(
        self,
        llm: BaseChatModel,
        schema_pydantic_base_model: type[BaseModel],
        # system_message: SystemMessage,
        max_retries: int = 3,
    ) -> BaseChatModel:
        """
        构造 structured_llm 的方法。

        这个实现默认:
            - 不对于 system-message 进行限制，不将 system-message 与 llm 提前绑定。
                - 但相关控制需要使用 structured_llm 的方法实现。

        Args:
            llm (BaseChatModel): 基础的用于推理的基座模型。需要具有结构化提取功能。
            schema_pydantic_base_model (BaseModel): 基于 pydantic 定义的 schema 。
            max_retries (int): 最大重试次数。基于 runnable 本身的实现。

        Returns:
            BaseChatModel: 被限制为仅会进行结构化输出的 structured_llm 。
        """
        structured_llm = llm.with_structured_output(
            schema=schema_pydantic_base_model,
        ).with_retry(
            stop_after_attempt=max_retries,
        )
        structured_llm = cast('BaseChatModel', structured_llm)
        return structured_llm

