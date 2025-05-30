"""
测试JsonOutputExtractor功能。

这是一个重要的工具类，未来任需要额外扩展。
"""

import pytest

from agnostic_utils.json_output_extractor import JsonOutputExtractor
from pydantic import BaseModel, Field


class _TestListDataClass(BaseModel):
    items: list[str] = Field(...)


class _TestDictDataClass(BaseModel):
    a: int
    b: str
    c: bool


_test_extract_json_from_str_cases = [
    (
        dict(
            raw_str='```json{"a": 1, "b": 2}```',
            index_to_choose=-1,
            json_loader_name='json-repair',
            schema_pydantic_base_model=None,
            schema_check_type='dict',
        ),
        {'a': 1, 'b': 2},
    ),
]


class TestJsonOutputExtractor:
    @pytest.mark.parametrize('inputs, expected', _test_extract_json_from_str_cases)
    def test_extract_json_from_str(self, inputs, expected):
        json_data = JsonOutputExtractor.extract_json_from_str(**inputs)
        assert json_data == expected

