"""
Sources:
    https://github.com/yuliu625/Yu-Agent-Development-Toolkit/blob/main/modules/langchain_toolkit/model_factory/local_llm_factory.py

References:
    None

Synopsis:
    使用本地进行响应的 LLM 。

Notes:
    可用的方法:
        - self-hosted openai-api: 本地运行 REST-API 服务，常见场景为本地部署 gateway 。
        - transformers 系列: 运行效率低，但是最灵活。有大量模型可以使用，并且可自定义方法。
        - ollama: langchain 集成度高，很方便。

    建议实践:
        - 不引入额外的类，仅使用 ollama 。
        - 本地大量部署模型，完全实现一个新的 LocalLLMFactory 。该文件的类仅做参考。
"""

from __future__ import annotations
from loguru import logger

# from langchain_ollama.chat_models import ChatOllama  # from langchain_community.chat_models import ChatOllama
# from langchain.llms import LlamaCpp
# from langchain.llms import GPT4All
from langchain_openai import ChatOpenAI

from pydantic import SecretStr

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class LocalLLMFactory:
    """
    使用本地 LLM 的工厂。
    """

    # ==== 使用 OpenAI API 兼容服务。 ====
    @staticmethod
    def create_openai_llm(
        base_url: str,
        api_key: str | SecretStr,
        model_name: str,
        temperature: float | None,
        max_tokens: int | None,
        logprobs: bool | None,
        # stream_options: dict,
        use_responses_api: bool | None,
        max_retries: int | None,
        model_configs: dict,
    ) -> ChatOpenAI:
        """
        使用相关推理框架，在本地运行模型。
        需要额外的工程实现模型部署。

        更好的做法:
            - 自动化脚本，以配置文件方式启动模型。
            - url 写在 .env 文件中，额外实现新的类，在必要是启动对应的模型。

        Args:
            base_url (str): 本地推理服务的地址。
            api_key (Union[str, SecretStr]): 调用服务使用的密钥。约定:
                - 以 environment variable 指定该值。
                - 该构建方法内部会处理为 SecretStr 。
            model_name (str): 模型的名称。
            temperature (float, optional):
            max_tokens (int, optional):
            logprobs (bool, optional):
            use_responses_api (bool, optional):
            max_retries (int, optional):
            model_configs (dict): 对于 ChatOpenAI 构造函数指定的 kwargs 。

        Returns:
            ChatOpenAI: langchain 中基础的 chat-model 。
        """
        logger.debug(f"Model configs: {model_configs}")
        llm = ChatOpenAI(
            base_url=base_url,  # 在绝大多数情况下，该字段需要修改。
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            # streaming=stream_options,
            use_responses_api=use_responses_api,
            max_retries=max_retries,
            api_key=api_key,  # trust me. I know what I am doing.
            **model_configs,
        )
        logger.info(f"Created {model_name}")
        return llm

    # ==== 常用方法。 ====
    @staticmethod
    def create_ollama_llm(
        model_name: str,
        reasoning: bool | None,
        temperature: float,
        num_predict: int | None,
        model_configs: dict,
    ) -> BaseChatModel:
        """
        使用 ollama 本地运行模型。

        最简单的方法，几乎没有迁移成本，并且支持的模型很多。

        注意:
            - model_name 的命名风格与远程 client 不同。

        Refactor:
            在 OpenAI client 中设置 base_url 为 http://127.0.0.1:11434/v1/ ，可实现统一调度。
            因此该方法可以不使用。

        ollama 可以自行管理:
            - 自启动和关闭模型。
            - 多 client 并保持线程安全。

        Args:
            model_name (str): 具体的模型。
                实际为 Literal ，在 https://ollama.com/ 查看。有命名空间，需要区别 checkpoint 等。
            reasoning:
            temperature:
            num_predict:
            model_configs (dict): 对于 ChatOllama 构造指定的 kwargs 。

        Returns:
            BaseChatModel: langchain 中可用于对话的 LLM 。
        """
        raise NotImplementedError("可直接使用 ollama 的 OpenAI-API ，减少不必要的调试和维护。")
        logger.debug(f"Model configs: {model_configs}")
        llm = ChatOllama(
            model=model_name,
            reasoning=reasoning,
            temperature=temperature,
            num_predict=num_predict,
            **model_configs,
        )
        logger.info(f"Created {model_name}")
        return llm

    # ==== 预留的方法。但不是最好的实践。 ====
    @staticmethod
    def create_huggingface_llm(
        *args, **kwargs,
    ) -> BaseChatModel:
        """
        使用基于 huggingface 上的 checkpoint 运行模型。

        仅适用于原型测试。不是很好的做法，大规模运行需要根据具体情况去实现。

        实现方法:
            - self-hosted: 使用 vllm 或 sglang 等推理引擎，高效进行推理。
            - microservice: 使用 litserve 等工具，大概基于 fast-api ，临时使用并调用针对性特殊方法。
        """
        raise NotImplementedError

