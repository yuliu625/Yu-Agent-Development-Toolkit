"""
Tests for openai_forge.openai_file_manager.py
"""

from __future__ import annotations
import pytest
from loguru import logger

from openai_forge.openai_file_id_and_name_mapping import (
    OpenAIFileIdAndNameMappingMethods,
)

from typing import TYPE_CHECKING, Sequence, Mapping
# if TYPE_CHECKING:


def _get_file_objects() -> Sequence[Mapping]:
    file_object = [

    ]
    return file_object


class TestOpenAIFileIdAndNameMappingMethods:
    @pytest.mark.parametrize(
        'file_objects', [
        _get_file_objects(),
    ])
    def test_collect_all_file_id(
        self,
        file_objects: Sequence[Mapping],
    ) -> None:
        file_ids = OpenAIFileIdAndNameMappingMethods.collect_all_file_id(
            file_objects=file_objects,
        )
        logger.info(f"File IDs: \n{file_ids}")

    @pytest.mark.parametrize(
        'file_objects', [
        _get_file_objects(),
    ])
    def test_collect_all_file_name(
        self,
        file_objects: Sequence[Mapping],
    ) -> None:
        file_names = OpenAIFileIdAndNameMappingMethods.collect_all_file_name(
            file_objects=file_objects,
        )
        logger.info(f"File names: \n{file_names}")

    @pytest.mark.parametrize(
        'file_objects', [
        _get_file_objects(),
    ])
    def test_collect_file_name_to_id_mapping(
        self,
        file_objects: Sequence[Mapping],
    ) -> None:
        name_to_id = OpenAIFileIdAndNameMappingMethods.collect_file_name_to_id_mapping(
            file_objects=file_objects,
        )
        logger.info(f"File name to id: \n{name_to_id}")

    @pytest.mark.parametrize(
        'file_objects', [
        _get_file_objects(),
    ])
    def test_collect_file_id_to_name_mapping(
        self,
        file_objects: Sequence[Mapping],
    ) -> None:
        id_to_name = OpenAIFileIdAndNameMappingMethods.collect_file_id_to_name_mapping(
            file_objects=file_objects,
        )
        logger.info(f"File names: \n{id_to_name}")

