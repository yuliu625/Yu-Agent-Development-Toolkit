"""
Sources:
    https://github.com/yuliu625/Yu-Agent-Development-Toolkit/content_processors/content_block_processor.py

References:
    None

Synopsis:
    对 content block 的处理和转换方法。

Notes:
    主要场景为:
        - VLM的HumanMessage.content的处理方法。
"""

from __future__ import annotations
from loguru import logger

import base64

from typing import TYPE_CHECKING, Literal
# if TYPE_CHECKING:


class ContentBlockProcessor:
    """
    VLM 输入 HumanMessage 需要的处理方法。

    多模态情况下，HumanMessage 的 content 字段需要是 list[dict] ，即 HumanMessage(content=[text_dict | image_dict]) 。
    使用该工具类处理得到的 dict ，还需要组合为一个 list 。
    """
    # ==== 主要方法。 ====
    @staticmethod
    def get_image_content_block_from_base64(
        base64_str: str,
        image_type: Literal['png'] = 'png',
    ) -> dict:
        """
        将原始 base64 编码过的图片转换为可与 VLM 交互的 dict 格式。

        Args:
            base64_str (str): 已经经过 base64 编码的图片。
            image_type (Literal['png']): 图片的类型。需要 VLM 支持，默认为 png 。

        Returns:
            dict: 添加了必要字段的 dict 。当前 content 中图片模态的内容。
        """
        image_content_dict = {
            'type': 'image',
            'source_type': 'base64',
            'mime_type': f'image/{image_type}',
            'data': base64_str,
        }
        return image_content_dict

    # ==== 主要方法。 ====
    @staticmethod
    def get_image_content_block_from_uri(
        uri: str,
        image_type: Literal['png'] = 'png',
    ) -> dict:
        """
        使用图片路径加载并转换图片为可与 VLM 交互的 dict 格式。

        Args:
            uri (str): 图片的路径。可以使用本地路径。
            image_type (Literal['png']): 图片的类型。需要 VLM 支持，默认为 png 。
                这里可以使用 pathlib 自动解析避免该字段传入，但是:
                    - uri 可能不含有图片类型的扩展名。
                    - 需要额外检测 VLM 是否支持图片类型。
                    - 如果需要自动识别，可在该工具类外额外写一个很简洁的方法。

        Returns:
            dict: 添加了必要字段的 dict 。当前 content 中图片模态的内容。
        """
        with open(uri, 'rb') as image_file:
            base64_str = base64.b64encode(image_file.read()).decode('utf-8')
        return ContentBlockProcessor.get_image_content_block_from_base64(
            base64_str=base64_str,
            image_type=image_type,
        )

    # ==== 主要方法。 ====
    @staticmethod
    def get_text_content_block(
        text: str,
    ) -> dict:
        """
        将原始文本转换为可与 VLM 交互的 dict 格式。

        只有文本模态的 message 并不需要这个方法。

        Args:
            text (str): 人类文本内容。

        Returns:
            dict: 添加了必要字段的 dict 。当前 content 中文本模态的内容。
        """
        text_content_dict = {
            'type': 'text',
            'text': text,
        }
        return text_content_dict

