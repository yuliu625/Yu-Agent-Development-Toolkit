"""
Sources:
    https://github.com/yuliu625/Yu-Agent-Development-Toolkit/blob/main/src/langchain_toolkit/content_processors/structured_data_extractor.py

References:
    None

Synopsis:
    从字符串中提取用 markdown-code-cell 包裹的 json 数据。

Notes:
    不同于 JsonOutputExtractor ，这个方法已经默认一定会提取结构化数据。
"""

from __future__ import annotations
from loguru import logger

import json
import json5
import json_repair
import re

from typing import TYPE_CHECKING, Literal, cast
if TYPE_CHECKING:
    from pydantic import BaseModel


class StructuredDataExtractor:
    """
    工具类，用于提取 LLM 的结构化输出。
    这个工具类是 JsonOutputExtractor 的简化，面对的场景是一定会去提取 structured-data 的情况。

    最终方法返回 BaseModel 或者 None 。None 无论是什么原因，都需要重新生成。

    流程为:
        raw_str (str) --re--> raw_json_str (str) --json.loads--> raw_structured_data (JSONReturnType)
        --schema--> structured_data (BaseModel)

    主要方法:
        - extract_structured_data_from_str: 封装所有操作的方法，需要指定相关参数。

    注意:
        - 最终结果的提取，dict和list不同:
            - dict: .model_dump()
            - list: .model_dump().items
    """

    # ====主要方法。====
    @staticmethod
    def extract_structured_data_from_str(
        raw_str: str,
        schema_pydantic_base_model: type[BaseModel],
        index_to_choose: int = -1,
        json_loader_name: Literal['json', 'json5', 'json-repair'] = 'json-repair',
        schema_check_type: Literal['dict', 'list'] = 'dict',
    ) -> BaseModel | None:
        """
        主要方法。从字符串格式中提取结构化数据。

        默认:
            - 从 markdown-code-cell 中提取 json 数据。

        实现:
            - 正则提取 markdown-code-cell 中的内容。默认提取最后一个。
            - 使用 json 相关库加载和转换数据。有多种加载工具，因为该工具类设计面对的是 LLM 的输出，默认使用 'json-repair' 。
            - 使用 pydantic 解析具体字段的正确性。静默判断。

        常见情况:
            - 提取并检验 dict 。
                输入 [raw_str, index_to_choose, json_loader_name, schema_pydantic_base_model, schema_check_type='dict']。
            - 提取并检验 list 。
                输入 [raw_str, index_to_choose, json_loader_name, schema_pydantic_base_model, schema_check_type='list']。
                这种情况需要注意 pydantic_base_model 的定义，约定字段 `items` 。

        Args:
            raw_str (str): LLM 输出的 str 部分。
            index_to_choose (int, optional): 选择提取的索引。可能会输出多个结果。默认提取最后一个。
            json_loader_name (Literal['json', 'json5', 'json-repair']): 加载 json 数据的方法。3 个加载包的区别是:
                - json: 最严格，需要完全符合 json 定义。
                - json5: 符合 js 的定义可以正常解析。
                - json-repair: 大概有 json 数据的结构，会尝试自动修复。
                默认选择 json-repair ，这样最节省 LLM 推理资源。需要 schema-check 后面会有进一步判断操作。
            schema_pydantic_base_model (type[BaseModel], optional): pydantic 定义的数据类。当有这个参数，会进行 structured-output 检测。
            schema_check_type (Literal['dict', 'list']): 检验 schema 的方法。2 种方式为 dict 或 list 。

        Returns:
            Optional[BaseModel]:
                - BaseModel: 正常解析。输出可用于处理的 structured output 。
                - None: 解析失败。
                    可能有多种原因，或许需要重试机制。(最简单，也是这个工具类的目的。)
                    正常解析也会遇到 None ，但是无论那种原因， None 都不可以用，需要再次生成。
        """
        # 正则匹配
        raw_json_str = StructuredDataExtractor.re_match(
            raw_str=raw_str, index_to_choose=index_to_choose
        )
        if not raw_json_str:
            return None  # 输出 1 。没有 json-output 。
        # 转换为 python 中的 structured-data
        raw_structured_data = StructuredDataExtractor.load_structured_data_from_raw_json_str(
            raw_json_str=raw_json_str, json_loader_name=json_loader_name
        )
        if not raw_structured_data:
            return None  # 输出 2 。structured-data 格式错误。
        # 获取由 pydantic 的 BaseModel 加载的 structured-data 。
        return StructuredDataExtractor.get_structured_data(
            raw_structured_data=raw_structured_data,
            schema_pydantic_base_model=schema_pydantic_base_model,
            schema_check_type=schema_check_type,
        )

    # ====基础方法。正则匹配。====
    @staticmethod
    def re_match(
        raw_str: str,
        index_to_choose: int = -1
    ) -> str | None:
        """
        正则查找 markdown-code-cell ，提取其中的结果。默认选择最后一个匹配项。

        Args:
            raw_str (str): 完全未处理的字符串结果
            index_to_choose (int): 选择提取的索引。
                可能会输出多个结果。默认提取最后一个。
                可进行自定义，但是一般 LLM 的输出限制会指定最后一个。

        Returns:
            Union[str, None]:
                - str: 正常提取的结果。
                - None: 完全没有匹配结果。
        """
        # 正则匹配。默认是 markdown-cell 中 json 数据。
        pattern = r'```json(.*?)```'
        matches = re.findall(pattern, raw_str, re.DOTALL)
        # 如果没有找到。一般在 prompt 中指定，就不会发生这种情况。
        if not matches:
            logger.warning("No JSON outputs.")
            return None  # 输出 1 。提取失败。
        # 提取结果。可能需要根据任务而定。
        raw_json_str: str = matches[index_to_choose]
        return raw_json_str  # 输出 2 。提取成功。

    # ====基础方法。加载json。====
    @staticmethod
    def load_structured_data_from_raw_json_str(
        raw_json_str: str,
        json_loader_name: Literal['json', 'json5', 'json-repair'] = 'json-repair',
    ) -> dict | list | None:
        """
        一些情况下，LLM 输出会带有奇怪格式。进行加载检验。

        Args:
            raw_json_str (str): 可能是 str 的 structured-data ，需要转换为 python 中的 structured-data 。
            json_loader_name (Literal['json', 'json5', 'json-repair']): 加载 json 数据的方法。3 个加载包的区别是:
                - json: 最严格，需要完全符合 json 定义。
                - json5: 符合 js 的定义可以正常解析。
                - json-repair: 大概有 json 数据的结构，会尝试自动修复。
                默认选择 json-repair ，这样最节省 LLM 推理资源。需要 schema-check 后面会有进一步判断操作。

        Returns:
            Union[Union[dict, list], None]:
                - Union[dict, list]: 转换成功。
                - None: 转换失败。
        """
        try:
            # 尝试进行转换。# 输出 2。成功转换。
            if json_loader_name == 'json':
                return json.loads(raw_json_str)
            elif json_loader_name == 'json5':
                return json5.loads(raw_json_str)
            elif json_loader_name == 'json-repair':
                return json_repair.loads(raw_json_str)
        except Exception as e:
            # 转换失败。打印错误，打印原始字符串。
            logger.error(e)
            logger.error(raw_json_str)
            logger.error("Fail to load structured data from raw_json_str.")
            return None  # 输出1。structured data格式错误。

    # ====基础方法。将python数据加载为pydantic结构化数据。====
    @staticmethod
    def get_structured_data(
        raw_structured_data: dict | list,
        schema_pydantic_base_model: type[BaseModel],
        schema_check_type: Literal['dict', 'list'] = 'dict',
    ) -> BaseModel | None:
        """
        对于已经可以加载的 json 数据再次加载为 pydantic 中的 data-model 。

        默认使用 pydantic ，原因在于:
            - 实际的严格检验。
            - 可以处理 number 和 bool 数据的自动转换，会很方便。

        Args:
            raw_structured_data (Union[dict, list]): json 数据，在这一步被处理为 python 中的 dict 或 list 的数据。
            schema_pydantic_base_model (Type[BaseModel]): pydantic 定义的数据类。
            schema_check_type (Literal['dict', 'list']): 进行加载的数据方法。

        Returns:
            Union[BaseModel, None]:
                - BaseModel: 通过检测。使用对应数据类型的方法获取 structured-data 。
                - None: 未通过检测。
            这个方法可以设计为输出 bool ，但为了和这个工具类中其他方法统一，设计为相同的输出方式。实际输出值仅用于逻辑判断。
        """
        # 冗余性检查，输入的数据类型需要和进行加载的数据类型一致。
        if schema_check_type == 'dict' and not isinstance(raw_structured_data, dict):
            return None
        if schema_check_type == 'list' and not isinstance(raw_structured_data, list):
            return None
        try:
            # 按照数据类别进行加载
            if schema_check_type == 'dict':
                raw_structured_data = cast('dict', raw_structured_data)
                return schema_pydantic_base_model(**raw_structured_data)  # 输出 2 。通过检测。使用 .model_dump() 方法获取。
            elif schema_check_type == 'list':
                raw_structured_data = cast('list', raw_structured_data)
                return schema_pydantic_base_model(items=raw_structured_data)  # 输出 2 。通过检测。使用 .model_dump().item 方法获取。
        except Exception as e:
            # 转换失败。打印错误，打印原始字符串。
            logger.error(e)
            logger.error(raw_structured_data)
            logger.error("Failed in schema.")
            return None  # 输出1。不符合 dataclass 定义。可能是字段，可能是数据类型。

