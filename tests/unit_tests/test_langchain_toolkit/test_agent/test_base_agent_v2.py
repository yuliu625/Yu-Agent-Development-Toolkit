"""
测试BaseAgent v2的情况。
"""

from __future__ import annotations
import pytest
import asyncio
from loguru import logger

from langchain_toolkit.agents.base_agent_v2 import BaseAgent
from langchain_toolkit.model_factory.local_llm_factory import LocalLLMFactory

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class Person(BaseModel):
    name: str = Field(description="The name of the person.")
    age: int = Field(description="The age of the person.")


class TestBaseAgent:
    @pytest.mark.parametrize(
        "main_llm, main_llm_system_message", [
        (LocalLLMFactory.create_ollama_llm(model_name='qwen2.5:1.5b', reasoning=None, temperature=0.7, num_predict=None, model_configs={}),
        SystemMessage(content="You are Bob.")),
    ])
    @pytest.mark.asyncio
    async def test_normal_agent(
        self,
        main_llm: BaseChatModel,
        main_llm_system_message: SystemMessage,
    ):
        agent = BaseAgent(
            main_llm=main_llm,
            main_llm_system_message=main_llm_system_message,
            main_llm_max_retries=3,
            is_need_structured_output=False,  # 这里设置为False进行简单测试。
            formatter_llm=None,
            schema_pydantic_base_model=None,
            formatter_llm_system_message=None,
            formatter_llm_max_retries=3,
        )
        response = await agent.a_call_llm_with_retry(
            messages=[
                main_llm_system_message,
                HumanMessage("Who are you?"),
            ],
        )
        logger.info(f"\nLLM Response: \n{response}")

    @pytest.mark.parametrize(
        "main_llm, main_llm_system_message, formatter_llm, schema_pydantic_base_model, formatter_llm_system_message", [
        (LocalLLMFactory.create_ollama_llm(model_name='qwen2.5:1.5b', reasoning=None, temperature=0.7, num_predict=None, model_configs={}),
         SystemMessage(content="You are Bob. You are 18 years old."),
         LocalLLMFactory.create_ollama_llm(model_name='qwen2.5:1.5b', reasoning=None, temperature=0.7, num_predict=None, model_configs={}),
         Person,
         SystemMessage(content="You extract person information from message."),)
    ])
    @pytest.mark.asyncio
    async def test_structured_output_agent(
        self,
        main_llm: BaseChatModel,
        main_llm_system_message: SystemMessage,
        formatter_llm: BaseChatModel | None,
        schema_pydantic_base_model: type[BaseModel],
        formatter_llm_system_message: SystemMessage | None,
    ):
        agent = BaseAgent(
            main_llm=main_llm,
            main_llm_system_message=main_llm_system_message,
            main_llm_max_retries=3,
            is_need_structured_output=True,  # 这里设置为True进行进阶测试。
            formatter_llm=formatter_llm,
            schema_pydantic_base_model=schema_pydantic_base_model,
            formatter_llm_system_message=formatter_llm_system_message,
            formatter_llm_max_retries=3,
        )
        response = await agent.a_call_llm_with_retry(
            messages=[
                main_llm_system_message,
                HumanMessage("Who are you? How old are you?"),
            ],
        )
        logger.info(f"\nAgent Response: \n{response}")
        logger.info(f"\nAgent Structured Output: \n{response.structured_output}")

