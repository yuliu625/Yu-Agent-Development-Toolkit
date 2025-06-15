"""
VLM的HumanMessage.content的处理方法。
"""

from __future__ import annotations

import base64

from typing import TYPE_CHECKING, Literal
# if TYPE_CHECKING:


class ContentBlockProcessor:
    """
    VLM输入HumanMessage需要的处理方法。

    多模态情况下，HumanMessage的content字段需要是list[dict]，即HumanMessage(content=[text_dict | image_dict])。
    使用该工具类处理得到的dict，还需要组合为一个list。
    """

    # ====主要方法。====
    @staticmethod
    def get_image_content_block_from_base64(
        base64_str: str,
        image_type: Literal['png'] = 'png',
    ) -> dict:
        """
        将原始base64编码过的图片转换为可与VLM交互的dict格式。

        Args:
            base64_str (str): 已经经过base64编码的图片。
            image_type (Literal['png']): 图片的类型。需要VLM支持，默认为png。

        Returns:
            dict: 添加了必要字段的dict。当前content中图片模态的内容。
        """
        image_content_dict = {
            'type': 'image',
            'source_type': 'base64',
            'mime_type': f'image/{image_type}',
            'data': base64_str,
        }
        return image_content_dict

    # ====主要方法。====
    @staticmethod
    def get_image_content_block_from_uri(
        uri: str,
        image_type: Literal['png'] = 'png',
    ) -> dict:
        """
        使用图片路径加载并转换图片为可与VLM交互的dict格式。

        Args:
            uri (str): 图片的路径。可以使用本地路径。
            image_type (Literal['png']): 图片的类型。需要VLM支持，默认为png。
                这里可以使用pathlib自动解析避免该字段传入，但是:
                    - uri可能不含有图片类型的扩展名。
                    - 需要额外检测VLM是否支持图片类型。
                    - 如果需要自动识别，可在该工具类外额外写一个很简洁的方法。

        Returns:
            dict: 添加了必要字段的dict。当前content中图片模态的内容。
        """
        with open(uri, "rb") as image_file:
            base64_str = base64.b64encode(image_file.read()).decode("utf-8")
        return ContentBlockProcessor.get_image_content_block_from_base64(
            base64_str=base64_str,
            image_type=image_type,
        )

    # ====主要方法。====
    @staticmethod
    def get_text_content_block(
        text: str,
    ) -> dict:
        """
        将原始文本转换为可与VLM交互的dict格式。

        只有文本模态的message并不需要这个方法。

        Args:
            text (str): 人类文本内容。

        Returns:
            dict: 添加了必要字段的dict。当前content中文本模态的内容。
        """
        text_content_dict = {
            'type': 'text',
            'text': text,
        }
        return text_content_dict

