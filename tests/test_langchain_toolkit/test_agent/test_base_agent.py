"""
测试BaseAgent的功能。
"""

from __future__ import annotations
import pytest

from langchain_toolkit.agents.base_agent_v1 import BaseAgent

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class TestBaseAgent:
    def test_base_agent(self, chat_prompt_template, llm):
        agent = BaseAgent(
            chat_prompt_template=chat_prompt_template,
            llm=llm,
        )
        response = agent.call_llm_with_retry(
            chat_prompt_template=chat_prompt_template,
            llm=llm,
            chat_history=[]
        )

