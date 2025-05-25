"""
来自llm_factory，但是并不使用openai兼容API，而是使用各家具体的client。
"""

import os
from dotenv import load_dotenv

from typing import TYPE_CHECKING, Annotated
if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class LLMFactory:
    """
    默认的全部以openai兼容API实现的LLM-factory。
    """
    def __init__(
        self,
        dotenv_path: str = None
    ):
        """
        不需要参数实现实例化的工厂。

        指定dotenv_path仅为项目相关的情况，例如使用项目的API_KEY。
        这些初始化操作可以更大的项目中删除并自行管理环境变量的导入。

        Args:
            dotenv_path: (str), .env文件的路径。不指定会自行寻找。
        """
        load_dotenv(dotenv_path=dotenv_path, override=True)

    @staticmethod
    def get_openai_llm(
        model: Annotated[str, "chatgpt系列模型的名字"] = 'gpt-4o-mini',
        **kwargs,
    ) -> BaseChatModel:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model,
            base_url=os.environ['OPENAI_API_BASE_URL'],
            api_key=os.environ['OPENAI_API_KEY'],
            **kwargs,
        )
        return llm

    @staticmethod
    def get_google_llm(
        model: Annotated[str, "gemini系列模型的名字"] = 'gemini-2.5-flash-preview-05-20',
        **kwargs,
    ) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model,
            base_url=os.environ['GEMINI_API_BASE_URL'],
            api_key=os.environ['GEMINI_API_KEY'],
            **kwargs,
        )
        return llm

    @staticmethod
    def get_anthropic_llm(
        model: Annotated[str, "claude系列模型的名字"] = 'claude-opus-4-20250514',
        **kwargs,
    ) -> BaseChatModel:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=model,
            base_url=os.environ['ANTHROPIC_API_BASE_URL'],
            api_key=os.environ['ANTHROPIC_API_KEY'],
            **kwargs,
        )
        return llm

    @staticmethod
    def get_dashscope_llm(
        model: Annotated[str, "qwen系列模型的名字"] = 'qwen-max',
        **kwargs,
    ) -> BaseChatModel:
        from langchain_community.chat_models.tongyi import ChatTongyi
        llm = ChatTongyi(
            model=model,
            base_url=os.environ['DASHSCOPE_API_BASE_URL'],
            api_key=os.environ['DASHSCOPE_API_KEY'],
            **kwargs,
        )
        return llm

    @staticmethod
    def get_deepseek_llm(
        model: Annotated[str, "deepseek系列模型的名字"] = 'deepseek-chat',
        **kwargs,
    ) -> BaseChatModel:
        from langchain_deepseek import ChatDeepSeek
        llm = ChatDeepSeek(
            model=model,
            base_url=os.environ['DEEPSEEK_API_BASE_URL'],
            api_key=os.environ['DEEPSEEK_API_KEY'],
            **kwargs,
        )
        return llm

