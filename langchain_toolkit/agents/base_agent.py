"""
基础的agent。

使用我构建的工具类JsonOutputExtractor，但是简化大量可设置的参数。

预期的派生类:
    - NormalAgent: 普通的对话agent。完全没有结构化数据相关的需求。
    - JsonAgent: 有结构化数据输出的agent。但并不提取结构化输出，仅以规范的方式与其他agent进行对话。
    - PydanticAgent: 强制输出符合schema的输出。需要使用结构化输出进行计算或指令。
"""

from __future__ import annotations

# 下面这个工具类是必要的，需要在具体项目中设定具体的导入路径。
from agnostic_utils.json_output_extractor import JsonOutputExtractor

from langchain_core.messages import AIMessage

from typing import TYPE_CHECKING, Literal, cast
if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AnyMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel


class BaseAgent:
    """
    基础的agent。具体结构化数据提取的功能。

    注意:
        - 没有chat-history维护的功能，仅持有system-prompt的状态。如果实现:
            - langchain: 在派生类中以属性维护。
            - langgraph: 维护图状态。
            2种实现派生类均需要实现chat-history的管理，可灵活自定义。
        - 多模态支持。熟悉命名中为llm，但也支持vlm。如果实现，需要相关的chat-history方法支持。

    具有功能:
        - 请求LLM进行生成。
        - 结构化输出校验和重试。
    """
    def __init__(
        self,
        chat_prompt_template: ChatPromptTemplate,
        llm: BaseChatModel,
        max_retries: int = 10,
        is_need_structured_output: bool = False,
        schema_pydantic_base_model: type[BaseModel] = None,
        schema_check_type: Literal['dict', 'list'] = 'dict',
    ):
        """
        必要的初始化参数。

        指定参数方法有:
            - 一般agent: 仅指定[chat_prompt_template, llm]。
            - 提取输出agent: 指定[chat_prompt_template, llm, is_need_structured_output=True]。
            - 强结构数据输出agent: 指定[chat_prompt_template, llm, is_need_structured_output=True, schema_pydantic_base_model, schema_check_type]

        Args:
            chat_prompt_template (ChatPromptTemplate): 构建的chat-prompt-template，一般仅包含system-prompt。
            llm (BaseChatModel): chat-model，可以生成内容。
            max_retries (int): 最大尝试生成次数。默认为10，更多的重试可能无意义，和模型的能力很有关系。
            is_need_structured_output (bool, optional): 是否需要结构化输出。如果不需要，仅一次响应。
            schema_pydantic_base_model (type[BaseModel], optional): 在需要结构化输出的情况下，进行dataclass检验。不指定，则不校验。
            schema_check_type (Literal['dict', 'list'], optional): 在需要结构化输出的情况下，进行dataclass检验的类型。常用为dict。
        """
        self._chat_prompt_template = chat_prompt_template
        self._llm = llm
        self._max_retries = max_retries
        self._is_need_structured_output = is_need_structured_output
        self._schema_pydantic_base_model = schema_pydantic_base_model
        self._schema_check_type = schema_check_type

    # ====最基础方法。====
    @staticmethod
    def call_llm(
        chat_prompt_template: ChatPromptTemplate,
        llm: BaseChatModel,
        chat_history: list[AnyMessage],
    ) -> AIMessage:
        """
        构建chain，进行内容生成。

        这是一个最常用的chain。独立构建是为了:
            - 自主维护chat-history。
            - 更好的控制LLM。

        注意:
            - 约定chat_history是以'chat_history'这个key传递。
            - 对于传入的chat_prompt_template，它是以system-message而不是以system-message-prompt-template初始化的。因为:
                - system-message更加安全。
                - system-message-prompt-template可以提前format。
                - chat_prompt_template可以以partial实现类似的效果，但复杂化问题。
                - 这个类默认是固定身份的agent，并处于确定的计算图。
            - 默认LLM的响应一定是AIMessage。

        Args:
            chat_prompt_template (ChatPromptTemplate): 构建的chat-prompt-template，一般仅包含system-prompt。
            llm (BaseChatModel): chat-model，可以生成内容。
            chat_history (list[AnyMessage]): 过去的对话记录。

        Returns:
            AIMessage: LLM的增量响应。如果包含tool-use的内容，会包含在AIMessage中。
        """
        llm_chain = chat_prompt_template | llm
        response = llm_chain.invoke(input={'chat_history': chat_history})
        response = cast('AIMessage', response)
        # assert isinstance(response, AIMessage)
        return response

    # ====最基础方法。====
    @staticmethod
    async def a_call_llm(
        chat_prompt_template: ChatPromptTemplate,
        llm: BaseChatModel,
        chat_history: list[AnyMessage],
    ) -> AIMessage:
        llm_chain = chat_prompt_template | llm
        response = await llm_chain.ainvoke(input={'chat_history': chat_history})
        response = cast('AIMessage', response)
        # assert isinstance(response, AIMessage)
        return response

    # ====主要方法。====
    def call_llm_with_retry(
        self,
        chat_history: list[AnyMessage],
    ) -> AIMessage:
        """
        请求LLM直至生成满足要求的结构化输出。

        需要使用:
            - JsonOutputExtractor: 已构建的工具类。
            - self.call_llm: 进行一般请求。
            - self._chat_prompt_template (ChatPromptTemplate): 构建的chat-prompt-template，一般仅包含system-prompt。
            - self._llm (BaseChatModel): chat-model，可以生成内容。
            - self._is_need_structured_output: 是否需要结构化输出。如果不需要，仅一次响应。
            - self._max_retries: 最大尝试生成次数。
            - self._schema_pydantic_base_model: 在需要结构化输出的情况下，进行dataclass检验。
            - self._schema_check_type: 在需要结构化输出的情况下，进行dataclass检验的类型。

        Args:
            chat_history (list[AnyMessage]): 过去的对话记录。

        Returns:
            AIMessage: LLM的增量响应。按照指定要求，符合不同要求的structured-output。
        """
        # 如果不需要结构化输出，得到一次请求的响应即可。
        if not self._is_need_structured_output:
            return self.call_llm(
                chat_prompt_template=self._chat_prompt_template,
                llm=self._llm,
                chat_history=chat_history,
            )
        # 如果需要结构化输出，在最大可重试次数内进行请求。
        for _ in range(self._max_retries):
            response = self.call_llm(
                chat_prompt_template=self._chat_prompt_template,
                llm=self._llm,
                chat_history=chat_history,
            )
            # 检测响应内容，是否符合结构化输出要求。
            if self.get_structured_output(raw_str=response.content):
                # 如果是有内容的，返回响应。
                return response

    # ====主要方法。====
    async def a_call_llm_with_retry(
        self,
        chat_history: list[AnyMessage],
    ) -> AIMessage:
        # 如果不需要结构化输出，得到一次请求的响应即可。
        if not self._is_need_structured_output:
            return await self.a_call_llm(
                chat_prompt_template=self._chat_prompt_template,
                llm=self._llm,
                chat_history=chat_history,
            )
        # 如果需要结构化输出，在最大可重试次数内进行请求。
        for _ in range(self._max_retries):
            response = await self.a_call_llm(
                chat_prompt_template=self._chat_prompt_template,
                llm=self._llm,
                chat_history=chat_history,
            )
            # 检测响应内容，是否符合结构化输出要求。
            if self.get_structured_output(raw_str=response.content):
                # 如果是有内容的，返回响应。
                return response

    # ====工具方法。====
    def get_structured_output(
        self,
        raw_str: str,
    ) -> dict | list | None:
        """
        提取结构化数据。用于条件判断和提取生成结果。

        Args:
            raw_str (str): 原始LLM输出的字符串。

        需要使用:
            - JsonOutputExtractor: 已构建的工具类。
            - self._schema_pydantic_base_model: 在需要结构化输出的情况下，进行dataclass检验。
            - self._schema_check_type: 在需要结构化输出的情况下，进行dataclass检验的类型。

        Returns:
            Union[Union[dict, list], None]:
                - Union[dict, list]: 提取出的结构化数据。
                - None: 没有提取出结构化数据。需要重试。
        """
        return JsonOutputExtractor.extract_json_from_str(
            raw_str=raw_str,
            index_to_choose=-1,
            json_loader_name='json-repair',
            schema_pydantic_base_model=self._schema_pydantic_base_model,
            schema_check_type=self._schema_check_type,
        )

    # ====冗余方法。====
    @staticmethod
    def get_chat_prompt_template(
        system_message: SystemMessage,
    ):
        """
        这是个冗余的方法。配套使用PromptTemplateLoader是直接可以加载chat_prompt_template的。
        """

