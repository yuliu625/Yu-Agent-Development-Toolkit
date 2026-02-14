"""
Sources:
    https://github.com/yuliu625/Yu-Agent-Development-Toolkit/openai_forge/openai_file_manager.py

References:

Synopsis:
    基于 OpenAI SDK 的文件管理方法。

Notes:
    封装了基础的增删改查方法。
    添加了批量处理方法。
"""

from __future__ import annotations
from loguru import logger

from openai import OpenAI

import os
from pathlib import Path

from typing import TYPE_CHECKING, Sequence
# if TYPE_CHECKING:


class OpenAIFileManager:
    @staticmethod
    def create_file(
        base_url: str,
        api_key: str,
        file_path: str | Path,
        purpose: str,
    ) -> dict:
        # 构建client。
        client = OpenAIFileManager.create_openai_client(
            base_url=base_url,
            api_key=api_key,
        )
        # 上传文件。
        file_object = client.files.create(
            file=file_path,
            purpose=purpose,
        )
        return file_object.model_dump()

    @staticmethod
    def create_files(
        base_url: str,
        api_key: str,
        file_paths: Sequence[str | Path],
        purpose: str,
    ) -> list[dict]:
        # 构建client。
        client = OpenAIFileManager.create_openai_client(
            base_url=base_url,
            api_key=api_key,
        )
        # 批量上传。
        file_object_results = []
        for file_path in file_paths:
            file_object = client.files.create(
                file=file_path,
                purpose=purpose,
            )
            file_object_results.append(file_object.model_dump())
        return file_object_results

    @staticmethod
    def retrieve_file(
        base_url: str,
        api_key: str,
        file_id: str,
    ) -> dict:
        # 构建client。
        client = OpenAIFileManager.create_openai_client(
            base_url=base_url,
            api_key=api_key,
        )
        # 通过 file-id 检索。
        file_object = client.files.retrieve(
            file_id=file_id,
        )
        return file_object.model_dump()

    @staticmethod
    def list_files(
        base_url: str,
        api_key: str,
    ) -> list[dict]:
        # 构建client。
        client = OpenAIFileManager.create_openai_client(
            base_url=base_url,
            api_key=api_key,
        )
        # 利用自动分页迭代器，循环获取。
        # all_files = []
        # for file in client.files.list():
        #     all_files.append(file.model_dump())
        all_files = list(client.files.list())
        all_files = [file.model_dump() for file in all_files]
        return all_files

    @staticmethod
    def delete_file(
        base_url: str,
        api_key: str,
        file_id: str,
    ) -> dict:
        # 构建client。
        client = OpenAIFileManager.create_openai_client(
            base_url=base_url,
            api_key=api_key,
        )
        # 删除文件。
        file_object = client.files.delete(
            file_id=file_id,
        )
        return file_object.model_dump()

    @staticmethod
    def delete_files(
        base_url: str,
        api_key: str,
        file_ids: Sequence[str],
    ) -> list[dict]:
        # 构建client。
        client = OpenAIFileManager.create_openai_client(
            base_url=base_url,
            api_key=api_key,
        )
        # 批量删除。
        file_object_results = []
        for file_id in file_ids:
            file_object = client.files.delete(
                file_id=file_id,
            )
            file_object_results.append(file_object.model_dump())
        return file_object_results

    # ==== 工具方法。 ====
    @staticmethod
    def create_openai_client(
        base_url: str,
        api_key: str,
    ) -> OpenAI:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        return client

