"""
测试StructuredDataExtractor工具类。
"""

from __future__ import annotations
import pytest

from src.content_processors.structured_data_extractor import StructuredDataExtractor
from tests.data.structured_data_extractor_cases import STRUCTURED_DATA_EXTRACTOR_CASES


# if TYPE_CHECKING:


class TestStructuredDataExtractor:
    @pytest.mark.parametrize('inputs, expected', STRUCTURED_DATA_EXTRACTOR_CASES)
    def test_extract_structured_data_from_str(self, inputs, expected):
        structured_data = StructuredDataExtractor.extract_structured_data_from_str(**inputs)
        assert structured_data == expected

