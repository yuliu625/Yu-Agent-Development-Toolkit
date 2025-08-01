"""
来自llm_factory，但是并不使用openai兼容API，而是使用各家具体的client。
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class SpecificLLMFactory:
    """
    以每个模型提供商的具体方法实现的LLM-factory。
    """
    # ====主要方法。====
    @staticmethod
    def create_llm(
        model_client: Literal['openai', 'google', 'anthropic', 'dashscope', 'deepseek', 'ollama'],
        model_name: str,
        model_configs: dict = None,
    ) -> BaseChatModel:
        """
        使用strategy-pattern封装的全部的方法。

        复杂构造仍需要传递对象参数。
        可以直接使用该工具类中其他方法。

        Args:
            model_client (Literal['openai', 'google', 'anthropic', 'dashscope', 'deepseek', 'ollama']): 模型的供应商。
            model_name (str): 具体模型的型号。
            model_configs (dict, optional): 对于ChatOpenAI构造函数指定的kwargs。

        Returns:
            BaseChatModel: langchain中可用于对话的LLM。
        """
        if model_client == 'openai':
            return SpecificLLMFactory.create_openai_llm(model_name=model_name, model_configs=model_configs)
        elif model_client == 'google':
            return SpecificLLMFactory.create_google_llm(model_name=model_name, model_configs=model_configs)
        elif model_client == 'anthropic':
            return SpecificLLMFactory.create_anthropic_llm(model_name=model_name, model_configs=model_configs)
        elif model_client == 'dashscope':
            return SpecificLLMFactory.create_dashscope_llm(model_name=model_name, model_configs=model_configs)
        elif model_client == 'deepseek':
            return SpecificLLMFactory.create_deepseek_llm(model_name=model_name, model_configs=model_configs)
        elif model_client == 'ollama':
            return SpecificLLMFactory.create_ollama_llm(model_name=model_name, model_configs=model_configs)

    @staticmethod
    def create_openai_llm(
        model_name: str,
        model_configs: dict = None,
    ) -> BaseChatModel:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model_name=model_name,
            # base_url=os.environ['OPENAI_API_BASE_URL'],
            api_key=os.environ['OPENAI_API_KEY_'],  # 修改默认环境变量名称，避免被偷偷使用OpenAI-API资源。
            **(model_configs or {}),
        )
        return llm

    @staticmethod
    def create_google_llm(
        model_name: str,
        model_configs: dict = None,
    ) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model_name,  # google这个kwarg只能指定为model。
            # base_url=os.environ['GEMINI_API_BASE_URL'],  # google这个kwarg不需要指定。
            api_key=os.environ['GEMINI_API_KEY'],
            transport='rest',  # 需要指定服务通信的网络协议为RESTful HTTP API，否则默认的grpc在使用代理服务时总是有问题。
            **(model_configs or {}),
        )
        return llm

    @staticmethod
    def create_anthropic_llm(
        model_name: str,
        model_configs: dict = None,
    ) -> BaseChatModel:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
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
    ) -> BaseChatModel:
        from langchain_community.chat_models.tongyi import ChatTongyi
        llm = ChatTongyi(
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
    ) -> BaseChatModel:
        from langchain_deepseek import ChatDeepSeek
        llm = ChatDeepSeek(
            model_name=model_name,
            api_base=os.environ['DEEPSEEK_API_BASE_URL'],
            api_key=os.environ['DEEPSEEK_API_KEY'],
            **(model_configs or {}),
        )
        return llm

    @staticmethod
    def create_ollama_llm(
        model_name: str,
        model_configs: dict = None,
    ) -> BaseChatModel:
        """
        使用ollama本地运行模型。

        最简单的方法，几乎没有迁移成本，并且支持的模型很多。

        注意:
            - model_name的命名风格与远程client不同。
。
        ollama可以自行管理:
            - 自启动和关闭模型。
            - 多client并保持线程安全。

        Args:
            model_name (str): 具体的模型。
                实际为Literal，在 https://ollama.com/ 查看。有命名空间，需要区别checkpoint等。
            model_configs (dict, optional): 对于ChatOllama构造指定的kwargs。

        Returns:
            BaseChatModel: langchain中可用于对话的LLM。
        """
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model=model_name,
            **(model_configs or {}),
        )
        return llm

