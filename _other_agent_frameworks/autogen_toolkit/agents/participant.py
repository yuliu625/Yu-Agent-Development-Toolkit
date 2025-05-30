"""
实验参与者的设定。

这是一个完全可以复用于任何任务的agent的模板。
"""

from game.global_setting import PARTICIPANT_MAX_RETRIES

from game.agents.participant.base_participant import BaseParticipant
from game.prompts import ParticipantPromptBuilder
from game.protocols import (
    ManagerRequest,
    ParticipantResponse,
)

from autogen_core import (
    RoutedAgent,
    MessageContext,
    TopicId,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    UserMessage,
)
from autogen_core.model_context import UnboundedChatCompletionContext, BufferedChatCompletionContext
from autogen_ext.models.openai import OpenAIChatCompletionClient

from pydantic import BaseModel
from typing import Type


# @type_subscription(topic_type='manager_request')
class Participant(BaseParticipant):
    """
    实验的参与者。
    可以进行：
        - 分析之前试验的结果。
        - 做出预期。
    """
    def __init__(
        self,
        description: str,
        model_client: OpenAIChatCompletionClient,
        prompt_builder: ParticipantPromptBuilder,  # 获取各种prompt的工厂，标准且更强大的格式化方法。
        model_context: UnboundedChatCompletionContext | BufferedChatCompletionContext,  # 模型的上下文，即状态。
        structured_output_format: Type[BaseModel] = None,  # 结构化输出格式的schema，默认使用pydantic。
        config: dict | None = None,  # required _PARTICIPANT_MAX_RETRIES
    ):
        super().__init__(
            description,
            model_client,
            prompt_builder,
            model_context,
            structured_output_format,
            config,
        )

    @message_handler
    async def on_manager_request(self, message: ManagerRequest, context: MessageContext) -> ParticipantResponse:
        print(f"{self.id.key}进行选择")
        # 维护模型上下文。
        await self._model_context.add_message(UserMessage(content=self.parse_manager_request(message.metadata), source='manager'))
        # 调用LLM进行响应。
        response = ""
        result = {}
        for i in range(PARTICIPANT_MAX_RETRIES):  # 这个retry不是模型本身的重试，而是因为不能够解析导致的重新生成。
            # print(f"第{i}次尝试生成。")
            # 进行最多10次尝试，需要返回的结果是符合通讯协议的。
            response = await self.request_llm(message, context)
            result = self._extract_json_from_response(response)
            if result:
                # 已经产生有效的结果，终止循环。
                break
        # 维护模型上下文。
        await self._model_context.add_message(AssistantMessage(content=response, source=self.id.key))
        # 发布选择结果。
        print(f"{self.id.key}的选择：\n{response}")
        # 广播发布信息。
        await self.publish_message(
            message=ParticipantResponse(
                content=response,
                metadata=dict(
                    participant_id=self.id.key,
                    **result,
                )
            ),
            topic_id=TopicId(type='participant_result', source=self.id.key)
        )
        # 返回manager可解析的协议数据。
        return ParticipantResponse(
            content=response,
            metadata=dict(
                participant_id=self.id.key,
                **result,
            )
        )

    def parse_manager_request(self, metadata: dict) -> str:
        """
        解析manager的结构化数据。

        Returns:
            使用了prompt_builder构建的可以输入给LLM的文本。
        """


class FakeParticipant(RoutedAgent):
    def __init__(self):
        super().__init__(description='用于测试的fake agent。')

    @message_handler
    async def on_manager_request(self, message: ManagerRequest, context: MessageContext) -> ParticipantResponse:
        return ParticipantResponse(
            content="这是一条来自fake participant的消息。",
            metadata=dict(
                participant_id=self.id.key,
                result={}
            )
        )

