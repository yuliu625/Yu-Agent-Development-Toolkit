"""
使用本地进行推理的LLM。
该文件的类仅做参考。

可用的方法:
    - transformers系列: 运行效率低，但是最灵活。有很多的模型可用使用，并且可用自定义很多方法。
    - ollama: langchain集成度高，很方便。
    - 本地openai-api： 本地运行REST-API服务，例如vllm。

建议实践:
    - 不引入额外的类，仅使用ollama。
    - 本地大量部署模型，完全实现一个新的LocalLLMFactory。该文件的类仅做参考。
"""

from __future__ import annotations
from loguru import logger

from langchain_ollama.chat_models import ChatOllama  # from langchain_community.chat_models import ChatOllama
# from langchain.llms import LlamaCpp
# from langchain.llms import GPT4All
from langchain_openai import ChatOpenAI

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class LocalLLMFactory:
    """
    使用本地LLM的工厂。
    """

    # ====常用方法。====
    @staticmethod
    def create_ollama_llm(
        model_name: str,
        reasoning: bool | None,
        temperature: float,
        num_predict: int | None,
        model_configs: dict,
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
            reasoning:
            temperature:
            num_predict:
            model_configs (dict): 对于ChatOllama构造指定的kwargs。

        Returns:
            BaseChatModel: langchain中可用于对话的LLM。
        """
        logger.debug(f"Model configs: {model_configs}")
        llm = ChatOllama(
            model=model_name,
            reasoning=reasoning,
            temperature=temperature,
            num_predict=num_predict,
            **model_configs,
        )
        logger.info(f"Created {model_name}.")
        return llm

    # ====使用OpenAI API兼容的服务。====
    @staticmethod
    def create_openai_llm(
        base_url: str,
        model_name: str,
        temperature: float | None,
        max_tokens: int | None,
        logprobs: bool | None,
        # stream_options: dict,
        use_responses_api: bool | None,
        max_retries: int | None,
        model_configs: dict,
    ) -> BaseChatModel:
        """
        使用相关推理框架，在本地运行模型。
        需要额外的工程实现模型部署。

        更好的做法:
            - 自动化脚本，以配置文件方式启动模型。
            - url写在.env文件中，额外实现新的类，在必要是启动对应的模型。

        Args:
            base_url (str): 本地推理服务的地址。
            model_name (str): 模型的名称。
            temperature:
            max_tokens:
            logprobs:
            use_responses_api:
            max_retries (int):
            model_configs (dict): 对于ChatOpenAI构造函数指定的kwargs。


        Returns:
            BaseChatModel: langchain中可用于对话的LLM。
        """
        logger.debug(f"Model configs: {model_configs}")
        llm = ChatOpenAI(
            base_url=base_url,  # 如果基于vllm，base_url是最重要的。
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            # streaming=stream_options,
            use_responses_api=use_responses_api,
            max_retries=max_retries,
            api_key='None',  # 类似vllm不需要校验api-key。
            **model_configs,
        )
        logger.info(f"Created {model_name}.")
        return llm

    # ====预留的方法。但不是最好的实践。====
    @staticmethod
    def create_huggingface_llm(
        *args, **kwargs,
    ) -> BaseChatModel:
        """
        使用基于huggingface上的checkpoint运行模型。

        仅适用于原型测试。不是很好的做法，大规模运行需要根据具体情况去实现。
        """
        raise NotImplementedError

