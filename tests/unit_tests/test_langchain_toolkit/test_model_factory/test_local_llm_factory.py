"""
测试本地模型的方法。
"""

from __future__ import annotations
import pytest
from loguru import logger

from langchain_toolkit.model_factory.local_llm_factory import LocalLLMFactory

from pydantic import BaseModel, Field

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class Person(BaseModel):
    name: str = Field(description="The name of the person.")
    age: int = Field(description="The age of the person.")


class TestLocalLLMFactory:
    @pytest.mark.parametrize(
        "model_name, reasoning, temperature, num_predict, model_configs", [
        ('qwen2.5:1.5b', None, 0.7, None, {}),
    ])
    def test_ollama_llm(
        self,
        model_name: str,
        reasoning: bool | None,
        temperature: float,
        num_predict: int | None,
        model_configs: dict,
    ):
        llm  = LocalLLMFactory.create_ollama_llm(
            model_name=model_name,
            reasoning=reasoning,
            temperature=temperature,
            num_predict=num_predict,
            model_configs=model_configs,
        )
        response = llm.invoke("Who are you?")
        logger.info(f"\nLLM Response: \n{response}")

    @pytest.mark.parametrize(
        "model_name, reasoning, temperature, num_predict, model_configs", [
        ('qwen2.5:1.5b', None, 0.7, None, {}),
    ])
    def test_ollama_llm_with_structured_output(
        self,
        model_name: str,
        reasoning: bool | None,
        temperature: float,
        num_predict: int | None,
        model_configs: dict,
    ):
        llm  = LocalLLMFactory.create_ollama_llm(
            model_name=model_name,
            reasoning=reasoning,
            temperature=temperature,
            num_predict=num_predict,
            model_configs=model_configs,
        )
        structured_llm = llm.with_structured_output(
            schema=Person,
        )
        response = structured_llm.invoke("Amy is 18 years old.")
        assert isinstance(response, Person)
        logger.info(f"\nLLM Response: \n{response}")

