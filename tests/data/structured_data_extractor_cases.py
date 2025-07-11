"""
StructuredDataExtractor的测试用例。

这个测试很复杂，因此数据数据构建是通过py文件。
"""

from pydantic import BaseModel, Field


class _TestListDataClass(BaseModel):
    items: list[str] = Field(...)


class _TestDictDataClass(BaseModel):
    a: int
    b: str
    c: bool


STRUCTURED_DATA_EXTRACTOR_CASES = [
    # dict-cases
    (
        dict(
            raw_str='```json{"a": 1, "b": "zx", "c": false}```',
            index_to_choose=-1,
            json_loader_name='json-repair',
            schema_pydantic_base_model=_TestDictDataClass,
            schema_check_type='dict',
        ),
        _TestDictDataClass(**{'a': 1, 'b': 'zx', 'c': False}),
    ),
    # list-cases
    (
        dict(
            raw_str='```json["a", "b", "c"]```',
            index_to_choose=-1,
            json_loader_name='json-repair',
            schema_pydantic_base_model=_TestListDataClass,
            schema_check_type='list',
        ),
        _TestListDataClass(items=['a', 'b', 'c']),
    ),
]

