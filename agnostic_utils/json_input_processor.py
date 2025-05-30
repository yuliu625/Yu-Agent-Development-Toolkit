"""
将python中的结构化数据转换为json格式的字符串。

使用场景为:
    - 将结构化数据传递给LLM。

我的工程规范:
    - 结构化参数以json格式传递，并且用markdown-code-cell显式说明。
"""

from __future__ import annotations

import json


class JsonInputProcessor:
    """
    工具类，将结构化数据转换为字符串格式。

    主要方法:
        - put_in_markdown: 主要方法，将结构化数据转换并放入markdown-code-cell中。
        - get_json_str_from_python_structured_data: 主要实现方法，和put_in_markdown区别仅不包裹markdown-code-cell。
    """

    # ====主要方法。====
    @staticmethod
    def put_in_markdown(
        original_structured_data: dict | list,
        # need_escape: bool = False  # 这个参数几乎被完全放弃了，但仅注释相关代码。
    ) -> str:
        """
        主要方法。将结构化数据自动转化为markdown cell中的字符串。

        Args:
            original_structured_data (Union[dict, list]): 输入的数据。默认为python的dict或list。dict最好为record形式。
            # need_escape (bool): 处理{}使得可以应用于f-string。标准处理不应该使用，因此默认为false。

        Returns:
            str: 转换完成的字符串。
        """
        # 转换为字符串。
        json_str = JsonInputProcessor.get_json_str_from_python_structured_data(original_structured_data)
        # 放进markdown中。
        result = JsonInputProcessor.wrap_in_markdown_code_cell(json_str)
        # 下面代码被注释是为了几乎没有可能遇到的兼容性问题。
        # 如果需要大括号处理，进行处理。
        # if need_escape:
        #     result = JsonInputProcessor._escape_braces(result)
        return result

    # ====基础方法之一。====
    @staticmethod
    def get_json_str_from_python_structured_data(
        original_structured_data: dict | list
    ) -> str:
        """
        使用json库将原本的结构化数据转换为json格式的字符串。

        数据本身在python中是可以运行的，因此默认是可以正常加载的。

        Args:
            original_structured_data (Union[dict, list]): python中已经是结构化数据的dict或list。可以传递额外的kwargs使用其他功能。

        Returns:
            str: 已经转换为json格式的字符串。
        """
        return json.dumps(original_structured_data, ensure_ascii=False)  # 由于中文的原因，需要指定ensure_ascii避免转换。

    # ====基础方法之一。====
    @staticmethod
    def wrap_in_markdown_code_cell(
        json_str: str
    ) -> str:
        """
        将已经转换好的 json 数据放入 markdown 的代码块中。

        需要使用 get_json_str_from_python_structured_data 方法进行转换，因为:
            - 使用 json 相关库加载可以进行一次简易的检查。(不是简单数据结构。)
            - 直接使用 f-string 会使得转换后的字符串中的引号均为单引号。

        Args:
            json_str (str): 已经转换为字符串的json数据。

        Returns:
            str: 字符串的结构化输入，包裹在 markdown-code-cell 中。
        """
        return f"```json\n{json_str}\n```"

    # ====已弃用。使用jinja2、PromptTemplate、一次性完成所有的format，不会遇到这种情况。====
    @staticmethod
    def _escape_braces(
        string: str
    ) -> str:
        """
        替换大括号。

        这个方法的目的是:
            - 为了 f-string 可以正常解析。
            - 为了 json 数据避免被转换。
        但是，在我使用 jinja2 文件，以及 langchain 的 PromptTemplate 后，这个方法基本不再被使用。
        """
        string = string.replace("{", "{{").replace("}", "}}")
        return string

