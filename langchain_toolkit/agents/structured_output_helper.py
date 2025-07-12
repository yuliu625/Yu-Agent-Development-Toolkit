"""
辅助进行结构化输出工具。

让LLM直接在分析完成后进行结构化输出可能的问题:
    - 有时会影响LLM的能力
    - 以重试机制实现，在长上下文成本可能昂贵。

让另一个LLM总结并进行结构化输出:
    - 异构agent，更高的灵活性。
    - 仅一条文本，价格低，任务简单。
    - 但是，提取的结果可能不是原始agent的本意。
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from pydantic import BaseModel


class StructuredOutputHelper:
    def __init__(
        self,
        llm: BaseChatModel,
        schema_pydantic_base_model: type[BaseModel],
    ):
        self._llm = llm
        self._schema_pydantic_base_model = schema_pydantic_base_model

    # ====主要方法。====
    def extract_structured_data(
        self,
    ):
        ...

    # ====主要方法。====
    def fix_structured_data(
        self,
    ):
        ...

