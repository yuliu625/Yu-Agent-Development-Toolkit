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
# if TYPE_CHECKING:


class Person(BaseModel):
    name: str = Field(description="The name of the person.")
    age: int = Field(description="The age of the person.")


class TestBaseAgent:
    @pytest.mark.parametrize()
    async def test_normal_agent(
        self,
    ):
        ...

    @pytest.mark.parametrize()
    async def test_structured_output_agent(
        self,
    ):
        ...

