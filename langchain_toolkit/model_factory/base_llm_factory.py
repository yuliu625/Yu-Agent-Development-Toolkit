"""
生成LLM的工厂。

以openai兼容API实现的LLM-factory。
可以基于这个factory进行扩展，以指定更加具体的模型，例如完全无参数获取LLM。

仅需:
```shell
pip install -U langchain-openai
```

如果需要:
    - 使用具体厂商的一些功能: 使用SpecificLLMFactory。
    - 使用本地模型: 使用SpecificLLMFactory，以及参考LocalLLMFactory具体去实现。
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

from typing import Literal


class BaseLLMFactory:
    """
    默认的全部以openai兼容API实现的LLM-factory。
    """
    # ====主要方法。====
    @staticmethod
    def create_llm(
        model_client: Literal['openai', 'google', 'anthropic', 'dashscope', 'deepseek'],
        model_name: str,
        model_configs: dict = None,
    ) -> ChatOpenAI:
        """
        使用strategy-pattern封装的全部的方法。

        复杂构造仍需要传递对象参数。
        可以直接使用该工具类中其他方法。

        Args:
            model_client (Literal['openai', 'google', 'anthropic', 'dashscope', 'deepseek']): 模型的供应商。
            model_name (str): 具体模型的型号。
            model_configs (dict, optional): 对于ChatOpenAI构造函数指定的kwargs。

        Returns:
            ChatOpenAI: langchain中可用于对话的LLM。
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
            base_url=os.environ['OPENAI_API_BASE_URL'],
            api_key=os.environ['OPENAI_API_KEY'],
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

