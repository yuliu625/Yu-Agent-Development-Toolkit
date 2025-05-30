"""
使用本地进行推理的LLM。

本地模型用transformers运行效率较低，并且不适用于Agent大规模推理的场景。
更好的构建方法是，根据OpenAI-API标准，使用相关工具构建REST-API服务，
"""

from __future__ import annotations

from langchain.llms import LlamaCpp
from langchain.llms import GPT4All
from langchain_community.chat_models import ChatOllama

from typing import TYPE_CHECKING, Annotated
if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class LocalLLMFactory:
    """
    使用本地LLM的工厂。
    """
    def __init__(self):
        ...

    @staticmethod
    def get_local_llm_by_url(
        url: str,
    ) -> BaseChatModel:
        ...

    @staticmethod
    def get_huggingface_llm_by_key(
        key: str,
    ) -> BaseChatModel:
        ...

    @staticmethod
    def get_ollama_llm_by_key(
        key: str,
    ) -> BaseChatModel:
        ...

