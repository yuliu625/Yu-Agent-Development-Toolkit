"""
Sources:

References:

Synopsis:
    构造可以进行结构化输出的LLM。

Notes:
"""

from __future__ import annotations
from loguru import logger

from typing import TYPE_CHECKING, cast
if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from pydantic import BaseModel


class StructuredLLMBuilder:
    @staticmethod
    def build_structured_llm(
        llm: BaseChatModel,
        schema_pydantic_base_model: type[BaseModel],
        # system_message: SystemMessage,
        max_retries: int = 3,
    ) -> BaseChatModel:
        """
        构造structured_llm的方法。

        这个实现默认:
            - 不对于system-message进行限制，不将system-message与llm提前绑定。
                - 但相关控制需要使用structured_llm的方法实现。

        Args:
            llm (BaseChatModel): 基础的用于推理的基座模型。需要具有结构化提取功能。
            schema_pydantic_base_model (BaseModel): 基于pydantic定义的schema。
            max_retries (int): 最大重试次数。基于runnable本身的实现。

        Returns:
            BaseChatModel: 被限制为仅会进行结构化输出的structured_llm。
        """
        structured_llm = llm.with_structured_output(
            schema=schema_pydantic_base_model,
        ).with_retry(
            stop_after_attempt=max_retries,
        )
        structured_llm = cast('BaseChatModel', structured_llm)
        return structured_llm

