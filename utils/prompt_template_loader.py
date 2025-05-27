"""
从指定路径加载prompt-template的方法。

我实际工程的规范:
    - 从文件导入。几乎不硬编码prompt-template。因为:
        - 直接写字符串面临换行等问题，写起来很丑。
        - 方便进行版本控制。
    - jinja2格式。不使用txt。因为:
        - 可以写逻辑。
        - 可以写注释。
    - PromptTemplate对象。不使用f-string。因为:
        - 更好的控制。例如partial。
    - 以_prompt_template命名结尾。区分prompt和prompt-template。
"""

from langchain.prompts import PromptTemplate

from pathlib import Path

from typing import Annotated


class PromptTemplateLoader:
    """
    从文件系统读取prompt-template的工具。

    可以在外层构建一个prompt-template-factory，实现完全用strategy-pattern控制prompt-template。
    """

    # ====最主要方法。====
    @staticmethod
    def load_prompt_template_from_j2(
        prompt_template_path: Annotated[str | Path, 'prompt-template所在的路径'],
    ) -> PromptTemplate:
        """
        从文件路径加载prompt-template的方法。

        Args:
            prompt_template_path: prompt-template在本地保存的路径。

        Returns:
            (PromptTemplate), 可以进行langchain中相关操作的prompt-template。
        """
        prompt_template = PromptTemplate.from_file(
            template_file=prompt_template_path,
            template_format='jinja2',  # 需要指定，否则解析方式为f-string。
            encoding='utf-8',  # 需要指定，否则解码中文有问题。
        )
        return prompt_template

    # ====已弃用。旧的我自己构建的从指定路径加载prompt-template的方法。====
    @staticmethod
    def load_prompt_template_from_txt(
        prompt_template_path: str
    ) -> PromptTemplate:
        prompt_template = PromptTemplate.from_file(
            template_file=prompt_template_path,
            template_format='f-string',
            encoding='utf-8',  # 需要指定，否则解码中文有问题。
        )
        return prompt_template

    # ====已弃用。仅加载txt文本的方法。后续需要使用f-string处理。====
    @staticmethod
    def load_original_txt(
        file_path: str | Path
    ) -> str:
        """
        从指定路径读取.txt文件的文本。

        Args:
            file_path: .txt文件的路径。

        Returns:
            (str), 读取的str文本。
        """
        file_path = Path(file_path)
        text = file_path.read_text(encoding='utf-8')
        return text

