"""
从字符串中提取用markdown代码块包裹的json结果。

一般面对的情况是：指定LLM进行结构化输出，需要从中提取结果。
"""

import json
import re

from typing import TYPE_CHECKING, Annotated, Type
if TYPE_CHECKING:
    from pydantic import BaseModel


class JsonOutputParser:
    """
    这是一个工具类，用于提取LLM的结构化输出。
    所有方法都是无状态的。
    """
    @staticmethod
    def extract_json_from_str(
        raw_str: str,
        schema_model: Annotated[Type[BaseModel], "进行schema检测的dataclass，如果不输入，则不会进行检测。"] = None,
    ) -> dict | list | None:
        """
        主要方法。从字符串格式中提取json格式的输出结果。
        主要使用的是正则方法进行提取，默认以及规范的做法是从markdown代码块中提取。

        Args:
            raw_str: LLM输出的str部分。
            schema_model: pydantic定义的数据类。当有这个参数，会进行structured output检测。

        Returns:
            dict | list 正常解析。输出可用于处理的structured output。
            None 解析失败。可能有多种原因，或许需要重试机制（最简单，也是这个类的目的。）。
        """
        # 正则匹配
        raw_data_str = JsonOutputParser.re_match(raw_str)
        if not raw_data_str:
            return None  # 输出1。没有structured output。
        # 转换为structured data
        raw_data = JsonOutputParser.check_json_format(raw_data_str)
        if not raw_data:
            return None  # 输出2。structured data格式错误。
        # schema检测。只有需要的时候才检测。
        if schema_model and not JsonOutputParser.check_schema(raw_data=raw_data, schema_model=schema_model):
            return None  # 输出3。需要schema检测，并且检测未通过。(not None，2个条件都为true。)
        # 通过所有的检测。
        return raw_data  # 输出4。不需要schema检测。或者需要schema检测，同时检测通过。

    @staticmethod
    def re_match(raw_str: str, index_to_choose: int = -1) -> str | None:
        """
        使用re进行查找。

        Args:
            raw_str: 完全未处理的字符串结果
            index_to_choose: 选择提取的索引。可能会输出多个结果。默认提取最后一个。

        Returns:
           str 正常提取。
           None 没有结果。
        """
        # 正则匹配。默认是markdown cell中json数据。
        pattern = r'```json(.*?)```'
        matches = re.findall(pattern, raw_str, re.DOTALL)
        # 如果没有找到。一般在prompt中指定，就不会发生这种情况。
        if not matches:
            print("无json输出。")
            return None  # 输出1。提取失败。
        # 提取结果。可能需要根据任务而定。
        raw_data_str: str = matches[index_to_choose]
        return raw_data_str  # 输出2。提取成功。

    @staticmethod
    def check_json_format(raw_data_str: str) -> dict | list | None:
        """
        一些情况下，LLM输出会带有奇怪格式。进行加载检验。

        Args:
            raw_data_str: 已经是structured data，但数据类型是str的输入。

        Returns:
            dict | list: 转换成功。
            None: 转换失败。
        """
        """
        TODO: 可以使用一些非严格的json工具，可以自动进行修正。
        """
        try:
            # 尝试进行转换。
            raw_data = json.loads(raw_data_str)
        except Exception as e:
            # 转换失败。打印错误，打印原始字符串。
            print(e)
            print(raw_data_str)
            print("未通过format检验。")
            return None  # 输出1。structured data格式错误。
        return raw_data  # 输出2。成功转换。

    @staticmethod
    def check_schema(raw_data: dict | list, schema_model: Type[BaseModel]) -> dict | list | None:
        """
        对于已经可以加载的json数据进行字段检验。
        默认使用pydantic，原因在于：
            - 实际的严格检验。
            - 可以处理number和bool数据的自动转换，会很方便。

        Args:
            raw_data: json数据，实际上这一步已经是python的dict的数据。
            schema_model: pydantic定义的数据类。

        Returns:
            dict | list: 通过检测。但是不进行dataclass加载，而是又外部代码实现。
            None: 未通过检测。
            可以设计未输出bool，但是为了和这个工具类中其他方法兼容，统一设置为相同的输出方式。
        """
        try:
            # dataclass定义检测，试转换。
            schema_model(**raw_data)
        except Exception as e:
            # 转换失败。打印错误，打印原始字符串。
            print(e)
            print(raw_data)
            print("未通过schema检验。")
            return None  # 输出1。不符合dataclass定义。可能是字段，可能是数据类型。
        return raw_data  # 输出2。通过检测。但是不进行额外处理。

    @staticmethod
    def delete_last_json(text: str) -> str:
        """
        使用正则方法，从原始字符串中删除最后一个markdown的json-cell。

        Args:
            text: (str), 原始文本。

        Returns:
            (str), 删除最后后一个markdown的json-cell的文本。
        """
        pattern = r'```json(.*?)```'
        matches = list(re.finditer(pattern, text, re.DOTALL))
        last = matches[-1]
        start, end = last.span()
        return text[:start] + text[end:]

