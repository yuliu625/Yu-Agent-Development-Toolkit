"""
Sources:
    https://github.com/yuliu625/Yu-Agent-Development-Toolkit/openai_forge/openai_file_id_and_name_mapping.py

References:

Synopsis:
    file-id 和 file-name 的映射方法。

Notes:
    本地 file_name 和 OpenAI client 上的 file_id 处理方法。
"""

from __future__ import annotations
from loguru import logger

from typing import TYPE_CHECKING, Sequence, Mapping
if TYPE_CHECKING:
    from openai.types import FileObject


class OpenAIFileIdAndNameMappingMethods:
    @staticmethod
    def collect_file_name_to_id_mapping(
        file_objects: Sequence[Mapping],
    ) -> dict[str, str]:
        name_to_id = {
            file_object['filename']: file_object['id']
            for file_object in file_objects
        }
        return name_to_id

    @staticmethod
    def collect_file_id_to_name_mapping(
        file_objects: Sequence[Mapping],
    ) -> dict[str, str]:
        id_to_name_mapping = {
            file_object['id']: file_object['filename']
            for file_object in file_objects
        }
        return id_to_name_mapping

    @staticmethod
    def collect_all_file_id(
        file_objects: Sequence[Mapping],
    ) -> list[str]:
        file_ids = [
            file_object['id']
            for file_object in file_objects
        ]
        return file_ids

    @staticmethod
    def collect_all_file_name(
        file_objects: Sequence[Mapping],
    ) -> list[str]:
        file_names = [
            file_object['filename']
            for file_object in file_objects
        ]
        return file_names

