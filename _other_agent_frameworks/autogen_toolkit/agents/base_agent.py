"""
实验参与者的设定。

这是一个完全可以复用于任何任务的agent的基础模板。
在这之上构建的agent只需要关注相关消息的通讯。
"""

from .utils import JsonOutputParser

from autogen_core import (
    RoutedAgent,
    MessageContext,
)
from autogen_core.models import (
    SystemMessage,
    LLMMessage,
)

from typing import TYPE_CHECKING, Type
if TYPE_CHECKING:
    from autogen_core.model_context import UnboundedChatCompletionContext, BufferedChatCompletionContext
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from pydantic import BaseModel


class BaseAgent(RoutedAgent):
    """
    实验agent的基础功能。

    默认是LLM-based agent，会对LLM进行请求。

    可以进行：
        - 维护自己的状态。
        - 返回结构化响应处理。
    """
    def __init__(
        self,
        description: str,
        model_client: OpenAIChatCompletionClient,
        prompt_builder: ParticipantPromptBuilder,  # 获取各种prompt的构建器，标准且更强大的格式化方法。
        model_context: UnboundedChatCompletionContext | BufferedChatCompletionContext,  # 模型的上下文，即状态。
        structured_output_format: Type[BaseModel] = None,  # 结构化输出格式的schema，默认使用pydantic。
        config: dict | None = None,
    ):
        super().__init__(description)
        # LLM相关。
        self._model_client = model_client  # LLM本身
        self._prompt_builder = prompt_builder  # 对话
        self._model_context = model_context  # 状态
        self._structured_output_format = structured_output_format  # 结构化输出格式
        self._config = config  # 预留设置字段。

        # 状态相关。
        self._system_prompts: list[LLMMessage] = self._get_system_prompts()

    async def request_llm(self, message, context: MessageContext) -> str:
        """
        对于LLM client进行请求。默认是有状态的。

        Return: 仅响应的字符串部分。
        """
        llm_result = await self._model_client.create(
            messages=self._system_prompts + await self._model_context.get_messages(),
            cancellation_token=context.cancellation_token,
        )
        response: str = llm_result.content
        # print(response)
        return response

    def _get_system_prompts(self) -> list[LLMMessage]:
        """
        获取system_prompt的方法。

        返回list形式，可以通过context相关的方法直接获取和处理。
        可以使用self.id.key来指定具体每个agent的身份。
        """
        system_prompt: str = self._prompt_builder.get_system_prompt(agent_id=self.id.key)
        return [SystemMessage(content=system_prompt)]

    def _get_inference_prompt_template(self, *args, **kwargs) -> str:
        """
        获得inference prompt template的方法。

        可以通过广播来实现，因此并不一定会使用。
        """
        inference_prompt: str = self._prompt_builder.get_inference_prompt(*args, **kwargs)
        return inference_prompt

    def _extract_json_from_response(self, response: str) -> dict | None:
        """这个方法使用我已经构建的复用工具来实现。"""
        if self._structured_output_format is None:
            return JsonOutputParser.extract_json_from_str(response)
        else:
            return JsonOutputParser.extract_json_from_str(
                response,
                schema_pydantic_base_model=self._structured_output_format,
            )

