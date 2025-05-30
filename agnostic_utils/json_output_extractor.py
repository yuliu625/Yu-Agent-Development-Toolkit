"""
从字符串中提取用markdown-code-cell包裹的json数据。

一般的使用场景是:
    - 指定LLM进行结构化输出，需要从中提取结果。

与langchain中的OutputParser的区别:
    - JsonOutputParser: 同样可以进行json数据的解析，同时支持json和markdown-code-cell中json这2种。我的实现是markdown-code-cell版本。
        但是，langchain中的实现较为简单，不能自定义相关功能。会默认提取第一个json，只能以最严格的json进行加载。
    - PydanticOutputParser: langchain中基于JsonOutputParser的派生类，更强大，但是对于pydantic的BaseModel的定义有限。
    - GuardrailsOutputParser: 调用LLM进行修复json，但是需要额外导入包，并且不是原本的LLM。
    - RetryWithErrorOutputParser: 重新请求，但是请求方式过于简单，不是原本的请求方法。
"""

from __future__ import annotations

import json
import json5
import json_repair
import re

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from pydantic import BaseModel


class JsonOutputExtractor:
    """
    工具类，用于提取LLM的结构化输出。

    所有方法均返回 JSONReturnType 或者 None。None无论是什么原因，都需要重新生成。

    流程为:
        raw_str (str) --re--> raw_json_str (str) --json.loads--> raw_structured_data (JSONReturnType)
        --schema--> structured_data (BaseModel)

    主要方法:
        - extract_json_from_str: 封装所有操作的方法，需要指定相关参数。
    """

    # ====主要方法。====
    @staticmethod
    def extract_json_from_str(
        raw_str: str,
        index_to_choose: int = -1,
        json_loader_name: Literal['json', 'json5', 'json-repair'] = 'json-repair',
        schema_pydantic_base_model: type[BaseModel] = None,
        schema_check_type: Literal['dict', 'list'] = 'dict',
    ) -> dict | list | None:
        """
        主要方法。从字符串格式中提取json格式的输出结果。

        默认:
            - 从markdown-code-cell中提取json数据。

        实现:
            - 正则提取markdown-code-cell中的内容。默认提取最后一个。
            - 使用json相关库加载和转换数据。有多种加载工具，因为该工具类设计面对的是LLM的输出，默认使用 'json-repair' 。
            - 使用pydantic解析具体字段的正确性。静默判断。

        常见情况:
            - 仅提取 json 数据。
                输入 [raw_str, index_to_choose, json_loader_name]。
            - 提取并检验dict。
                输入 [raw_str, index_to_choose, json_loader_name, schema_pydantic_base_model, schema_check_type='dict']。
            - 提取并检验list。
                输入 [raw_str, index_to_choose, json_loader_name, schema_pydantic_base_model, schema_check_type='list']。
                这种情况需要注意pydantic_base_model的定义，约定字段`items`。

        Args:
            raw_str (str): LLM输出的str部分。
            index_to_choose (int, optional): 选择提取的索引。可能会输出多个结果。默认提取最后一个。
            json_loader_name (Literal['json', 'json5', 'json-repair']): 加载json数据的方法。3个加载包的区别是:
                - json: 最严格，需要完全符合json定义。
                - json5: 符合js的定义可以正常解析。
                - json-repair: 大概有json数据的结构，会尝试自动修复。
                默认选择json-repair，这样最节省LLM推理资源。需要schema-check后面会有进一步判断操作。
            schema_pydantic_base_model (type[BaseModel], optional): pydantic定义的数据类。当有这个参数，会进行structured-output检测。
            schema_check_type (Literal['dict', 'list']): 检验schema的方法。2种方式为dict或list。

        Returns:
            Union[Union[dict, list], None]:
                - Union[dict, list]: 正常解析。输出可用于处理的structured output。
                - None: 解析失败。
                    可能有多种原因，或许需要重试机制。(最简单，也是这个工具类的目的。)
                    正常解析也会遇到None，但是无论那种原因，None都不可以用，需要再次生成。
        """
        # 正则匹配
        raw_json_str = JsonOutputExtractor.re_match(
            raw_str=raw_str, index_to_choose=index_to_choose
        )
        if not raw_json_str:
            return None  # 输出1。没有json-output。
        # 转换为python中的structured-data
        raw_structured_data = JsonOutputExtractor.load_structured_data_from_raw_json_str(
            raw_json_str=raw_json_str, json_loader_name=json_loader_name
        )
        if not raw_structured_data:
            return None  # 输出2。structured-data格式错误。
        # schema检测。只有需要的时候才检测。
        if schema_pydantic_base_model:
            if schema_check_type == 'dict' and not JsonOutputExtractor.check_dict_schema(
                raw_dict_structured_data=raw_structured_data,
                schema_pydantic_base_model=schema_pydantic_base_model,
            ):
                return None  # 输出3。需要schema检测，并且检测未通过。(not None，2个条件都为True。)
            elif schema_check_type == 'list' and not JsonOutputExtractor.check_list_schema(
                raw_list_structured_data=raw_structured_data,
                schema_pydantic_base_model=schema_pydantic_base_model,
            ):
                return None  # 输出3。需要schema检测，并且检测未通过。(not None，2个条件都为True。)
        # 通过所有的检测。或者不需要schema检测。
        return raw_structured_data  # 输出4。不需要schema检测。或者，需要schema检测，同时检测通过。

    # ====基础方法。正则匹配。====
    @staticmethod
    def re_match(
        raw_str: str,
        index_to_choose: int = -1
    ) -> str | None:
        """
        正则查找markdown-code-cell，提取其中的结果。默认选择最后一个匹配项。

        Args:
            raw_str (str): 完全未处理的字符串结果
            index_to_choose (int): 选择提取的索引。
                可能会输出多个结果。默认提取最后一个。
                可进行自定义，但是一般LLM的输出限制会指定最后一个。

        Returns:
            Union[str, None]:
                - str: 正常提取的结果。
                - None: 完全没有匹配结果。
        """
        # 正则匹配。默认是markdown-cell中json数据。
        pattern = r'```json(.*?)```'
        matches = re.findall(pattern, raw_str, re.DOTALL)
        # 如果没有找到。一般在prompt中指定，就不会发生这种情况。
        if not matches:
            print("无json输出。")
            return None  # 输出1。提取失败。
        # 提取结果。可能需要根据任务而定。
        raw_json_str: str = matches[index_to_choose]
        return raw_json_str  # 输出2。提取成功。

    # ====基础方法。加载json。====
    @staticmethod
    def load_structured_data_from_raw_json_str(
        raw_json_str: str,
        json_loader_name: Literal['json', 'json5', 'json-repair'] = 'json-repair',
    ) -> dict | list | None:
        """
        一些情况下，LLM输出会带有奇怪格式。进行加载检验。

        Args:
            raw_json_str (str): 可能是str的structured-data，需要转换为python中的structured-data。
            json_loader_name (Literal['json', 'json5', 'json-repair']): 加载json数据的方法。3个加载包的区别是:
                - json: 最严格，需要完全符合json定义。
                - json5: 符合js的定义可以正常解析。
                - json-repair: 大概有json数据的结构，会尝试自动修复。
                默认选择json-repair，这样最节省LLM推理资源。需要schema-check后面会有进一步判断操作。

        Returns:
            Union[Union[dict, list], None]:
                - Union[dict, list]: 转换成功。
                - None: 转换失败。
        """
        try:
            # 尝试进行转换。# 输出2。成功转换。
            if json_loader_name == 'json':
                return json.loads(raw_json_str)
            elif json_loader_name == 'json5':
                return json5.loads(raw_json_str)
            elif json_loader_name == 'json-repair':
                return json_repair.loads(raw_json_str)
        except Exception as e:
            # 转换失败。打印错误，打印原始字符串。
            print(e)
            print(raw_json_str)
            print("未通过format检验。")
            return None  # 输出1。structured data格式错误。

    # ====基础方法。检查schema。====
    @staticmethod
    def check_dict_schema(
        raw_dict_structured_data: dict,
        schema_pydantic_base_model: type[BaseModel],
    ) -> dict | None:
        """
        对于已经可以加载的json数据进行字段检验。

        静默检查，如果需要转换由外部工具实现。

        默认使用pydantic，原因在于：
            - 实际的严格检验。
            - 可以处理number和bool数据的自动转换，会很方便。

        Args:
            raw_dict_structured_data (dict): json数据，在这一步被处理为python中的dict的数据。
            schema_pydantic_base_model (Type[BaseModel]): pydantic定义的数据类。

        Returns:
            Union[dict, None]:
                - dict: 通过检测。但是不进行dataclass加载，而是由外部逻辑实现。
                - None: 未通过检测。
            这个方法可以设计为输出bool，但为了和这个工具类中其他方法统一，设计为相同的输出方式。实际输出值仅用于逻辑判断。
        """
        if not isinstance(raw_dict_structured_data, dict):
            return None
        try:
            # dataclass定义检测，试转换。
            schema_pydantic_base_model(**raw_dict_structured_data)
        except Exception as e:
            # 转换失败。打印错误，打印原始字符串。
            print(e)
            print(raw_dict_structured_data)
            print("未通过schema检验。")
            return None  # 输出1。不符合dataclass定义。可能是字段，可能是数据类型。
        return raw_dict_structured_data  # 输出2。通过检测。但是不进行额外处理。

    # ====基础方法。检查schema。====
    @staticmethod
    def check_list_schema(
        raw_list_structured_data: list,
        schema_pydantic_base_model: type[BaseModel],
    ) -> list | None:
        """
        对于已经可以加载的json数据进行字段检验。对于list。

        我知道可以使用pydantic的RootModel简化这个方法实现。
        但是为了和check_dict_schema统一，以及避免额外转换逻辑。
        约定规范减少复杂性。

        注意: 这里的pydantic-base-model需要具体的定义形式。例如:
            ```python
            class ListToCheck(BaseModel):
                items: list[str] = Field(..., min_items=3, max_items=5)
            ```
            约定只有一个`items`字段，list长度判断由pydantic中定义Field实现。

        Args:
            raw_list_structured_data (list): 原始的结构化数据。
            schema_pydantic_base_model (type[BaseModel]): pydantic定义的数据类。

        Returns:
            Union[list, None]:
                - list: 通过检测。
                - None: 未通过检测。
            这个方法可以设计为输出bool，但为了和这个工具类中其他方法统一，设计为相同的输出方式。实际输出值仅用于逻辑判断。
        """
        if not isinstance(raw_list_structured_data, list):
            return None
        try:
            # dataclass定义检测，试转换。
            schema_pydantic_base_model(items=raw_list_structured_data)
        except Exception as e:
            # 转换失败。打印错误，打印原始字符串。
            print(e)
            print(raw_list_structured_data)
            print("未通过schema检验。")
            return None  # 输出1。不符合dataclass定义。可能是字段，可能是数据类型。
        return raw_list_structured_data  # 输出2。通过检测。但是不进行额外处理。

    # ====暂未添加的方法。====
    @staticmethod
    def delete_last_json(
        text: str
    ) -> str:
        """
        使用正则方法，从原始字符串中删除最后一个markdown的json-cell。

        Args:
            text (str): 原始文本。

        Returns:
            str: 删除最后后一个markdown的json-cell的原始文本。
        """
        pattern = r'```json(.*?)```'
        matches = list(re.finditer(pattern, text, re.DOTALL))
        last = matches[-1]
        start, end = last.span()
        return text[:start] + text[end:]

