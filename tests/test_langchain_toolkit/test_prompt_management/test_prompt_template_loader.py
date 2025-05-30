"""
测试PromptTemplateLoader的功能。
"""

from __future__ import annotations
import pytest

from langchain_toolkit.prompt_management.prompt_template_loader import PromptTemplateLoader

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
    from langchain_core.messages import SystemMessage


_prompt_template_cases = [
    (r"tests\data\prompt_template_case.j2", "1"),
]


class TestPromptTemplateLoader:
    @pytest.mark.parametrize('inputs, expected', _prompt_template_cases)
    def test_load_prompt_template_from_j2(self, inputs, expected):
        prompt_template = PromptTemplateLoader.load_prompt_template_from_j2(inputs)
        print(prompt_template)
        assert isinstance(prompt_template, PromptTemplate)
        prompt = prompt_template.format()
        assert isinstance(prompt, str)

    @pytest.mark.parametrize('inputs, expected', )
    def test_load_chat_prompt_template_from_j2(self, inputs, expected):
        chat_prompt_template = PromptTemplateLoader.load_chat_prompt_template_from_j2(inputs)
        print(chat_prompt_template)
        assert isinstance(chat_prompt_template, ChatPromptTemplate)

    @pytest.mark.parametrize('inputs, expected', )
    def test_load_system_message_prompt_template_from_j2(self, inputs, expected):
        system_message_prompt_template = PromptTemplateLoader.load_system_message_prompt_template_from_j2(inputs)
        print(system_message_prompt_template)
        assert isinstance(system_message_prompt_template, SystemMessagePromptTemplate)
        system_message_prompt = system_message_prompt_template.format()
        assert isinstance(system_message_prompt, SystemMessage)

