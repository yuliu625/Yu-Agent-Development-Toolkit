"""
测试JsonOutputExtractor功能。

这是一个重要的工具类，未来任需要额外扩展。
"""

from __future__ import annotations
import pytest

from tests.data.json_output_extractor_cases import JSON_OUTPUT_EXTRACTOR_CASES
from agnostic_utils.json_output_extractor import JsonOutputExtractor

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class TestJsonOutputExtractor:
    @pytest.mark.parametrize('inputs, expected', JSON_OUTPUT_EXTRACTOR_CASES)
    def test_extract_json_from_str(self, inputs, expected):
        json_data = JsonOutputExtractor.extract_json_from_str(**inputs)
        assert json_data == expected

