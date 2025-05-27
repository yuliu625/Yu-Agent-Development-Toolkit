"""
从字符串中提取用markdown-code-cell包裹的json数据。

一般的使用场景是:
    - 指定LLM进行结构化输出，需要从中提取结果。
"""

import json
import json5
import json_repair
import re

from typing import TYPE_CHECKING, Annotated, Type, Literal
if TYPE_CHECKING:
    from pydantic import BaseModel


class JsonOutputParser:
    """
    工具类，用于提取LLM的结构化输出。
    """

    # ====主要方法====
    @staticmethod
    def extract_json_from_str(
        raw_str: str,
        index_to_choose: int = -1,
        json_loader: Literal['json', 'json5', 'json-repair'] = 'json-repair',
        schema_model: Annotated[Type[BaseModel], "进行schema检测的dataclass，如果不输入，则不会进行检测。"] = None,
    ) -> dict | list | None:
        """
        主要方法。从字符串格式中提取json格式的输出结果。

        默认:
            - 从markdown-code-cell中提取json数据。
        实现:
            - 正则提取markdown-code-cell中的内容。默认提取最后一个。
            - 使用json相关库加载和转换数据。有多种加载工具，因为该工具类设计面对的是LLM的输出，默认使用 'json-repair' 。
            - 使用pydantic解析具体字段的正确性。静默判断。

        Args:
            raw_str (str): LLM输出的str部分。
            index_to_choose (int): 选择提取的索引。可能会输出多个结果。默认提取最后一个。
            json_loader (Literal['json', 'json5', 'json-repair']): 加载json数据的方法。3个加载包的区别是:
                - json: 最严格，需要完全符合json数据的定义。
                - json5: 符合js的定义可以正常解析。
                - json-repair: 大概有json数据的结构，会尝试自动修复。
                默认选择json-repair，这样最节省LLM推理资源。后面会有schema-check再进一步判断。
            schema_model (type[BaseModel]): pydantic定义的数据类。当有这个参数，会进行structured output检测。

        Returns:
            Union[dict, list]: 正常解析。输出可用于处理的structured output。
            None: 解析失败。
                可能有多种原因，或许需要重试机制。(最简单，也是这个工具类的目的。)
                正常解析也会遇到None，但是无论那种原因，None都不可以用，需要再次生成。
        """
        # 正则匹配
        raw_data_str = JsonOutputParser.re_match(raw_str=raw_str, index_to_choose=index_to_choose)
        if not raw_data_str:
            return None  # 输出1。没有structured output。
        # 转换为structured data
        raw_data = JsonOutputParser.check_json_format(raw_data_str=raw_data_str, json_loader=json_loader)
        if not raw_data:
            return None  # 输出2。structured data格式错误。
        # schema检测。只有需要的时候才检测。
        if schema_model and not JsonOutputParser.check_schema(raw_data=raw_data, schema_model=schema_model):
            return None  # 输出3。需要schema检测，并且检测未通过。(not None，2个条件都为true。)
        # 通过所有的检测。
        return raw_data  # 输出4。不需要schema检测。或者需要schema检测，同时检测通过。

    @staticmethod
    def re_match(
        raw_str: str,
        index_to_choose: int = -1
    ) -> str | None:
        """
        正则查找markdown-code-cell。

        Args:
            raw_str (str): 完全未处理的字符串结果
            index_to_choose (int): 选择提取的索引。可能会输出多个结果。默认提取最后一个。

        Returns:
           str: 正常提取。
           None: 没有结果。
        """
        # 正则匹配。默认是markdown-cell中json数据。
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
    def check_json_format(
        raw_data_str: str,
        json_loader: Literal['json', 'json5', 'json-repair'] = 'json-repair',
    ) -> dict | list | None:
        """
        一些情况下，LLM输出会带有奇怪格式。进行加载检验。

        Args:
            raw_data_str (str): 已经是structured data，但数据类型是str的输入。
            json_loader (Literal['json', 'json5', 'json-repair']):

        Returns:
            Union[dict, list]: 转换成功。
            None: 转换失败。
        """
        try:
            # 尝试进行转换。# 输出2。成功转换。
            if json_loader == 'json':
                return json.loads(raw_data_str)
            elif json_loader == 'json5':
                return json5.loads(raw_data_str)
            elif json_loader == 'json-repair':
                return json_repair.loads(raw_data_str)
        except Exception as e:
            # 转换失败。打印错误，打印原始字符串。
            print(e)
            print(raw_data_str)
            print("未通过format检验。")
            return None  # 输出1。structured data格式错误。

    @staticmethod
    def check_schema(
        raw_data: dict | list,
        schema_model: Type[BaseModel]
    ) -> dict | list | None:
        """
        对于已经可以加载的json数据进行字段检验。

        默认使用pydantic，原因在于：
            - 实际的严格检验。
            - 可以处理number和bool数据的自动转换，会很方便。

        Args:
            raw_data (Union[dict, list]): json数据，实际上这一步已经是python的dict的数据。
            schema_model (Type[BaseModel]): pydantic定义的数据类。

        Returns:
            Union[dict, list]: 通过检测。但是不进行dataclass加载，而是又外部代码实现。
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
    def delete_last_json(
        text: str
    ) -> str:
        """
        使用正则方法，从原始字符串中删除最后一个markdown的json-cell。

        Args:
            text (str): 原始文本。

        Returns:
            str: 删除最后后一个markdown的json-cell的文本。
        """
        pattern = r'```json(.*?)```'
        matches = list(re.finditer(pattern, text, re.DOTALL))
        last = matches[-1]
        start, end = last.span()
        return text[:start] + text[end:]

