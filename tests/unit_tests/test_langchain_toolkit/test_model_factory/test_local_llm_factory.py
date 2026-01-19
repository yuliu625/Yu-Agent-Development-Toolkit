"""
测试本地模型的方法。
"""

from __future__ import annotations
import pytest
from loguru import logger

from langchain_toolkit.model_factory.local_llm_factory import LocalLLMFactory

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


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

