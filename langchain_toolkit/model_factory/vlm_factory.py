"""
具有视觉能力的VLM的factory。
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

from typing import Annotated


class VLMFactory:
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
    def get_dashscope_vlm(
        model_name: Annotated[str, "qwen系列模型的名字"] = 'qwen-vl-max-latest',
        **kwargs,
    ):
        """
        获得qwen的VLM。
        由于现有的各种agent-framework对于VLM不完全支持，这里专门去使用。

        Returns:
            langchain中的MultiModalLLM。这里是qwen-vl-max，我默认使用这个。
        """
        dashscope_multi_modal_llm = ChatOpenAI(
            model_name=model_name,
            api_key=os.environ['DASHSCOPE_API_KEY'],
            vl_high_resolution_images=True,  # 因为文档的缺失，不是很确定这个参数是否有效。为了冗余，会在每次请求时额外再指定。
            **kwargs,
        )
        return dashscope_multi_modal_llm

