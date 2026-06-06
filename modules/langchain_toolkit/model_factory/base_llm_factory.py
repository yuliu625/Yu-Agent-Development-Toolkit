"""
Sources:
    https://github.com/yuliu625/Yu-Agent-Development-Toolkit/blob/main/modules/langchain_toolkit/model_factory/base_llm_factory.py

References:
    None

Synopsis:
    生成 LLM 的工厂。

Notes:
    以 openai 兼容 API 实现的 LLM-factory 。
    可以基于这个 factory 进行扩展，以指定更加具体的模型，例如完全无参数获取 LLM 。

    仅需:
    ```shell
    pip install -U langchain-openai
    ```

    如果需要:
        - 使用具体厂商的一些功能: 使用 SpecificLLMFactory 。
        - 使用本地模型: 使用 SpecificLLMFactory ，以及参考 LocalLLMFactory 具体去实现。

    未来改动: 由于科研导向需求，该工具未来做出以下改动:
        - 中间件: 自行维护各种 API 过于费力，未来考虑使用如 LiteLLM ，仅维护配置文件，不再进行具体实现。
        - 中间商: 考虑支付部分比例服务器，使用中间商统一服务。
"""

from __future__ import annotations
from loguru import logger

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

from typing import TYPE_CHECKING, Literal
# if TYPE_CHECKING:


class BaseLLMFactory:
    """
    默认的全部以 openai 兼容 API 实现的 LLM-factory 。
    """
    # ====主要方法。====
    @staticmethod
    def create_llm(
        model_client: Literal['openai', 'google', 'anthropic', 'dashscope', 'deepseek'],
        model_name: str,
        model_configs: dict = None,
    ) -> ChatOpenAI:
        """
        使用 strategy-pattern 封装的全部的方法。

        复杂构造仍需要传递对象参数。
        可以直接使用该工具类中其他方法。

        Args:
            model_client (Literal['openai', 'google', 'anthropic', 'dashscope', 'deepseek']): 模型的供应商。
            model_name (str): 具体模型的型号。
            model_configs (dict, optional): 对于 ChatOpenAI 构造函数指定的 kwargs 。

        Returns:
            ChatOpenAI: langchain 中可用于对话的 LLM 。
        """
        if model_client == 'openai':
            return BaseLLMFactory.create_openai_llm(model_name=model_name, model_configs=model_configs)
        elif model_client == 'google':
            return BaseLLMFactory.create_google_llm(model_name=model_name, model_configs=model_configs)
        elif model_client == 'anthropic':
            return BaseLLMFactory.create_anthropic_llm(model_name=model_name, model_configs=model_configs)
        elif model_client == 'dashscope':
            return BaseLLMFactory.create_dashscope_llm(model_name=model_name, model_configs=model_configs)
        elif model_client == 'deepseek':
            return BaseLLMFactory.create_deepseek_llm(model_name=model_name, model_configs=model_configs)

    @staticmethod
    def create_openai_llm(
        model_name: str,
        model_configs: dict = None,
    ) -> ChatOpenAI:
        llm = ChatOpenAI(
            model_name=model_name,
            # base_url=os.environ['OPENAI_API_BASE_URL'],
            api_key=os.environ['OPENAI_API_KEY_'],
            **(model_configs or {}),
        )
        return llm

    @staticmethod
    def create_google_llm(
        model_name: str,
        model_configs: dict = None,
    ) -> ChatOpenAI:
        llm = ChatOpenAI(
            model_name=model_name,
            base_url=os.environ['GEMINI_API_BASE_URL'],
            api_key=os.environ['GEMINI_API_KEY'],
            **(model_configs or {}),
        )
        return llm

    @staticmethod
    def create_anthropic_llm(
        model_name: str,
        model_configs: dict = None,
    ) -> ChatOpenAI:
        llm = ChatOpenAI(
            model_name=model_name,
            base_url=os.environ['ANTHROPIC_API_BASE_URL'],
            api_key=os.environ['ANTHROPIC_API_KEY'],
            **(model_configs or {}),
        )
        return llm

    @staticmethod
    def create_dashscope_llm(
        model_name: str,
        model_configs: dict = None,
    ) -> ChatOpenAI:
        llm = ChatOpenAI(
            model_name=model_name,
            base_url=os.environ['DASHSCOPE_API_BASE_URL'],
            api_key=os.environ['DASHSCOPE_API_KEY'],
            **(model_configs or {}),
        )
        return llm

    @staticmethod
    def create_deepseek_llm(
        model_name: str,
        model_configs: dict = None,
    ) -> ChatOpenAI:
        llm = ChatOpenAI(
            model_name=model_name,
            base_url=os.environ['DEEPSEEK_API_BASE_URL'],
            api_key=os.environ['DEEPSEEK_API_KEY'],
            **(model_configs or {}),
        )
        return llm

