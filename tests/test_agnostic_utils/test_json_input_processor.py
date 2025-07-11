"""
测试JsonInputProcessor的运行情况。
"""

from __future__ import annotations
import pytest

from agnostic_utils.json_input_processor import JsonInputProcessor
from typing import TYPE_CHECKING
# if TYPE_CHECKING:


_test_put_in_markdown_cases = [
    ([1, 2, 3], "```json\n[1, 2, 3]\n```"),
    ({'a': 1, 'b': 2}, '```json\n{"a": 1, "b": 2}\n```'),
]

_test_get_json_str_from_python_structured_data_cases = [
    ([1, 2, 3], "[1, 2, 3]"),
    ({'a': 1, 'b': 2}, '{"a": 1, "b": 2}'),
]


class TestJsonInputProcessor:
    @pytest.mark.parametrize('inputs, expected', _test_put_in_markdown_cases)
    def test_put_in_markdown(self, inputs, expected):
        markdown_str = JsonInputProcessor.put_in_markdown(inputs)
        assert markdown_str == expected

    @pytest.mark.parametrize('inputs, expected', _test_get_json_str_from_python_structured_data_cases)
    def test_get_json_str_from_python_structured_data(self, inputs, expected):
        json_str = JsonInputProcessor.get_json_str_from_python_structured_data(inputs)
        assert json_str == expected

