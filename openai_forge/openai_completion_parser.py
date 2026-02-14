"""
Sources:
    https://github.com/yuliu625/Yu-Agent-Development-Toolkit/openai_forge/openai_completion_parser.py

References:

Synopsis:
    从 OpenAI client 响应中解析需要的字段。

Notes:
    解析存在2种情况:
        - Completion object: 直接从响应中进行解析。
        - JSON: 加载缓存的结果，然后进行解析。
    所有方法均实现了2种解析方法。
"""

from __future__ import annotations
from loguru import logger

from typing import TYPE_CHECKING, Mapping, overload
if TYPE_CHECKING:
    from openai.types import Completion


class OpenAICompletionParser:
    @overload
    @staticmethod
    def extract_content(
        completion: Completion,
    ) -> str:
        ...

    @overload
    @staticmethod
    def extract_content(
        completion: Mapping,
    ) -> str:
        ...

    @staticmethod
    def extract_content(
        completion: Completion | Mapping,
    ) -> str:
        if isinstance(completion, Completion):
            content = completion.choices[0]['message']['content']
            return content
        elif isinstance(completion, Mapping):
            content = completion['choices'][0]['message']['content']
            return content
        raise TypeError("completion must be Completion or Mapping.")

