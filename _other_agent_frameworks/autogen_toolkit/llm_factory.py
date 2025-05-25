"""
Model-client的工厂。

我封装了autogen中创建LLM client的方法，完全无参数获取模型。
后续扩展实验需要增加工厂的产品。
"""

from autogen_ext.models.openai import OpenAIChatCompletionClient

import os
from enum import Enum


class QwenModelName(Enum):
    qwen_15 = 'qwen2.5-1.5b-instruct'
    qwen_max = 'qwen-max'
    qwen_plus = 'qwen-plus'
    qwen_turbo = 'qwen-turbo'


class DeepseekModelName(Enum):
    deepseek_chat = 'deepseek-chat'
    deepseek_reasoner = 'deepseek-reasoner'


class ModelClientFactory:
    @staticmethod
    def get_qwen(
        model: str = 'qwen2.5-1.5b-instruct'
    ):
        model_client = OpenAIChatCompletionClient(
            model=model,
            base_url=os.environ['DASHSCOPE_API_BASE_URL'],
            api_key=os.environ['DASHSCOPE_API_KEY'],
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": False,
                "family": 'unknown',
            },
        )
        return model_client

    @staticmethod
    def get_deepseek(
        model: str = 'deepseek-reasoner'
    ):
        model_client = OpenAIChatCompletionClient(
            model=model,
            base_url=os.environ['DEEPSEEK_API_BASE_URL'],
            api_key=os.environ['DEEPSEEK_API_KEY'],
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": 'r1',
            },
        )
        return model_client

