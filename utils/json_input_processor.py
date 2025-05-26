"""
结构化数据处理为LLM可处理的str。

用于需要LLM进行分析决策的场景。
"""

import json


class JsonInputProcessor:
    """
    这是一个工具类，用于处理结构化数据为字符串。
    所有方法都是无状态的。
    """
    @staticmethod
    def put_in_markdown(input_structured_data: dict | list, need_escape: bool = False) -> str:
        """
        主要方法。将结构化数据自动转化为markdown cell中的字符串。

        Args:
            input_structured_data: 输入的数据。默认为python的dict或list。dict最好为record形式。
            need_escape: 处理{}使得可以应用于f-string。标准处理不应该使用，因此默认为false。

        Returns:
            转换完成的字符串。
        """
        # 转换为字符串。
        json_str = JsonInputProcessor.serialize_structured_data_to_json(input_structured_data)
        # 放进markdown中。
        result = JsonInputProcessor.wrap_in_markdown(json_str)
        # 如果需要大括号处理，进行处理。
        if need_escape:
            result = JsonInputProcessor.escape_braces(result)
        return result

    @staticmethod
    def serialize_structured_data_to_json(input_structured_data: dict | list) -> str:
        """
        使用json库进行转换。

        数据本身在python中是可以运行的，因此默认是可以正常加载的。
        """
        return json.dumps(input_structured_data, ensure_ascii=False)  # 由于中文的原因，需要指定ensure_ascii避免转换。

    @staticmethod
    def wrap_in_markdown(json_str: str) -> str:
        """
        将已经转换好的json数据放入markdown的代码块中。

        Args:
            json_str: 字符串的json.不可以是dict，否则会有一个很小的问题是: 引号是单引号。

        Returns:
            字符串的结构化输入。
        """
        return f"```json\n{json_str}\n```"

    @staticmethod
    def escape_braces(string: str) -> str:
        """
        替换大括号。

        主要为了使得f-string可以正常解析。
        在我使用基于jinja2的模板以及langchain的PromptTemplate之后，这个方法基本不再被使用。
        """
        string = string.replace("{", "{{").replace("}", "}}")
        return string

