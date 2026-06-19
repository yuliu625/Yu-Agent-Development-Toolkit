"""
Sources:
    https://github.com/yuliu625/Yu-Agent-Development-Toolkit/blob/main/src/langchain_toolkit/agents/base_agent_v2.py

References:
    None

Synopsis:
    基础的 agent 。

Notes:
    V2:
        - 设计思路: 不限制模型的输出，结构化信息提取由具有 function-calling 模型独立执行。
        - features:
            - system-message 分离: system-message 不再作为 agent 实现上的固有属性，而是可以独立修改的。这有助于:
                - context-engineering: 可以直接将 system-prompt 进行卸载，从而修改 agent 身份。(更高效的 MAS 。)
                - batch-processing: 所有 context 可独立整合，批量处理。(在这个实现下，不用担心错误和重试。)
            - 结构化输出分离: 由另一具有结构化输出模型执行对应数据提取。
                - 优势:
                    - 自由输出: 无 system-message 限制，模型完全自主决定输出内容，数据总是会被正常识别。
                    - 短 context: 避免执行重试的长 context ，结构化输出只是对于一条 BaseMessage 。
                - 可能存在的问题:
                    - 不一致: 结构输出模型未能有效提取主模型输出。( 2 个模型是同一个基座模型可以解决这个问题。)
            - 简化语法: 我不再使用复杂的 langchain 语法。包括:
                - llm 和 messages 分离: 不再将 chat-prompt-template 与 llm 绑定，而是简单使用 list 完全控制所有 context 。
                    - llm 完全无状态，并且适用于修改和异步和多线程编程。
                    - 不再使用 ChatPromptTemplate ，仅使用 list[AnyMessage] ，显式控制全部的全部 context 。

    可能会增加的机制:
        - 回溯: 当 main-llm 输出出现问题，formatter-llm 无法正确提取信息，可以执行重试。(但该结构设计为更大的 agent-flow 会更好。)
"""

from __future__ import annotations
from loguru import logger

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING, cast
if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AnyMessage, SystemMessage, AIMessage
    from pydantic import BaseModel


class BaseAgentResponse(BaseModel):
    """
    当前文件BaseAgent中定义的输出格式。可根据具体项目需求修改。
    """

    ai_message: AIMessage = Field(
        description="原始的LLM返回的AIMessage",
    )
    structured_output: BaseModel | None = Field(
        description="根据BaseAgent的设置，提取的结构化输出。",
    )


class BaseAgent:
    """
    基础的 agent 。具体结构化数据提取的功能。

    Notes:
        - 没有 context 维护的功能，完全不持有 message 相关的状态。如果实现:
            - langchain: 在派生类中以属性维护。
            - langgraph: 维护图状态。
            2种实现派生类均需要实现 context 的管理，可灵活自定义。
        - 多模态支持。熟悉命名中为 llm ，但也支持 vlm 。如果实现，需要相关的 context 方法支持。

    具有功能:
        - 请求 LLM 进行生成。
        - 结构化输出校验和重试。

    预期使用方法:
        - 继承该超类，指定 BaseAgent 类型，仅管理 context ，使用 a_call_llm_with_retry 方法。
        - 如果需要，对于具体的 LLM ，重写 a_call_llm 方法。

    可能存在的优化:
        - 用 pydantic 定义 a_call_llm_with_retry 的输出。
    """

    def __init__(
        self,
        main_llm: BaseChatModel,
        main_llm_system_message: SystemMessage,
        main_llm_max_retries: int = 3,
        is_need_structured_output: bool = False,
        formatter_llm: BaseChatModel | None = None,
        schema_pydantic_base_model: type[BaseModel] = None,
        formatter_llm_system_message: SystemMessage | None = None,
        formatter_llm_max_retries: int = 3,
    ):
        """
        必要的初始化参数。

        指定参数方法有:
            - 一般 agent : 仅指定 main_llm 相关参数。
            - 强结构数据输出 agent : 设置 is_need_structured_output=True ，并需要额外指定 formatter_llm 相关参数。

        Args:
            main_llm (BaseChatModel): 任意的 llm ，可以生成内容。(甚至是 runnable 即可。)
            main_llm_system_message (SystemMessage): main_llm 的指令。
            main_llm_max_retries (int): 最大尝试生成次数。默认为 3 。
            is_need_structured_output (bool, optional): 是否需要结构化输出。如果不需要，返回的输出中 structured_output 为 None 。
            formatter_llm (BaseChatModel, optional): 有 with_structured_output 实现的 llm ，会被用于 structured_llm 的构造。
            schema_pydantic_base_model (type[BaseModel], optional): 在需要结构化输出的情况下，进行 dataclass 检验。(这里的 docstring 对 formatter 很重要。)
            formatter_llm_system_message (SystemMessage, optional): formatter 的指令。(有常见通用指令，也可以具体自定义。)
            formatter_llm_max_retries (int): 最大尝试生成次数。默认为 3 。
        """
        # main llm
        self._main_llm = main_llm
        self._main_llm_system_message = main_llm_system_message
        self._main_llm_max_retries = main_llm_max_retries
        # formatter llm
        self._is_need_structured_output = is_need_structured_output
        self._formatter_llm_system_message = formatter_llm_system_message  # 这个参数应该作为 BaseAgent 的内部属性，调用主要方法自动使用。
        self._formatter_llm_max_retries = formatter_llm_max_retries  # 这个参数应该作为 BaseAgent 的内部属性，调用主要方法自动使用。
        # build structured llm
        self._structured_llm = None
        if is_need_structured_output:
            # 判断参数合法性。
            if not (
                formatter_llm is not None
                and schema_pydantic_base_model is not None
                and formatter_llm_system_message is not None
            ):
                raise ValueError("需要结构化输出，formatter 相关的参数不能为 None 。")
            # 在需要结构化输出时构建，直接用实例化传入的参数构造 structured llm 。
            self._structured_llm = self._build_structured_llm(
                llm=formatter_llm,
                schema_pydantic_base_model=schema_pydantic_base_model,
                max_retries=formatter_llm_max_retries,
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
    async def a_call_llm_with_retry(
        self,
        messages: list[AnyMessage],
    ) -> BaseAgentResponse:
        """
        请求 LLM 直至生成满足要求的结构化输出。

        States:
            self.a_call_llm: 进行一般请求。
            self.get_structured_output: 提取结构化输出的方法。根据该类的实例设置，条件调用。
            self._is_need_structured_output (bool): 是否需要结构化输出。如果不需要，仅一次响应。
            self._formatter_system_message (SystemMessage): 如果进行结构化提取，formatter 的指令。
            self._main_llm_max_retries (int): main_llm 最大尝试生成次数。
            self._formatter_llm_max_retries (int): formatter_llm 最大尝试生成次数。

        Args:
            messages (list[AnyMessage]): 全部的 messages 。
                包括system-message，所有的 context 都由外部调用方法管理(e.g. self.process_state)。

        Returns:
            dict[str, AIMessage | None]: LLM 的增量响应。按照指定要求，符合不同要求的 structured-output 。
        """
        # 获取 main_llm 的输出。
        # 自实现简单重试机制，避免网络问题。
        response = None
        for _ in range(self._main_llm_max_retries):
            try:
                response = await self.a_call_llm(
                    llm=self._main_llm,
                    messages=messages,
                )
                break
            except Exception as e:
                logger.error(e)
        if response is None:
            raise RuntimeError("main llm 达到最大重试次数。")
        # 如果不需要结构化输出，直接返回响应结果。
        if not self._is_need_structured_output:
            logger.debug(f"Type of response: {type(response)}")
            return BaseAgentResponse(
                ai_message=response,
                structured_output=None,
            )  # 输出1: 仅输出 ai_message ，没有 structured_output 。
        # 如果需要结构化输出，在最大可重试次数内进行请求。
        else:
            structured_output = None
            for _ in range(self._formatter_llm_max_retries):
                try:
                    structured_output = await self.get_structured_output(
                        raw_str=response.content,
                        structured_llm=self._structured_llm,
                        formatter_system_message=self._formatter_llm_system_message,
                    )
                    break
                except Exception as e:
                    logger.error(e)
            if structured_output is None:
                raise RuntimeError("formatter llm 达到最大重试次数。")
            return BaseAgentResponse(
                ai_message=response,
                structured_output=structured_output,
            )  # 输出2: 仅输出 ai_message ，同时提供 structured_output 。

    # ==== 主要方法。 ====
    async def a_call_llm(
        self,
        llm: BaseChatModel,
        messages: list[AnyMessage],
    ) -> AIMessage:
        """
        使用 context ，通过 llm 进行内容生成。

        相比较 V1 的实现，该实现取消了 ChatPromptTemplate 的构造，并且在函数内部不再构造 chain 。

        Notes:
            - system-message 自我管理: 对于这个实现，调用该方法的函数需要在 messages 中自行控制 system-message 。

        Args:
            llm (BaseChatModel): chat-model，可以生成内容。
            messages (list[AnyMessage]): 全部的 messages 。

        Returns:
            AIMessage: LLM 的增量响应。如果包含 tool-use 的内容，会包含在 AIMessage 中。
        """
        # assert isinstance(messages[0], SystemMessage)  # 断言第一个 message 类型，非必要，部分推理框架有默认配置。
        response = await llm.ainvoke(input=messages)
        response = cast('AIMessage', response)
        # assert isinstance(response, AIMessage)
        return response

    # ==== 主要方法。 ====
    async def get_structured_output(
        self,
        raw_str: str,
        structured_llm: BaseChatModel,
        formatter_system_message: SystemMessage,
    ) -> BaseModel:
        """
        提取结构化数据。用于条件判断和提取生成结果。

        States:
            _structured_llm (BaseChatModel): 构造好的可以进行结构化提取的 llm 。
            _formatter_system_message (SystemMessage): formatter_llm 的指令。
        这里显式以调用的方法说明，以说明参与的对象。
        但 structured_output 应作为该类的内部必要方法，应该不需要外部设置。

        Args:
            raw_str (str): 原始 LLM 输出的字符串。
            structured_llm (BaseChatModel): 已经绑定了目标 schema 的 llm 。
            formatter_system_message (SystemMessage): 对 formatter_llm 的指令 system-message 。

        Returns:
            BaseModel: 基于初始定义 schema 的 pydantic-base-model 。
        """
        response = await structured_llm.ainvoke(
            input=[
                formatter_system_message,
                HumanMessage(raw_str),
            ],
        )
        return response

    # ==== 工具方法。 ====
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

