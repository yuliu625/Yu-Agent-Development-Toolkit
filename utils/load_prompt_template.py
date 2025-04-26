"""
从指定路径加载prompt-template的方法。
"""

from langchain.prompts import PromptTemplate

from pathlib import Path
from typing import Annotated


def load_prompt_template(
    prompt_template_path: Annotated[str | Path, 'prompt-template所在的路径'],
) -> PromptTemplate:
    """
    从文件路径加载prompt-template的方法。

    我的工程的默认:
        - prompt-template全部由文件读入。从而使得prompt可以完全配置化，和代码分离。
        - jinja2格式。jinja2有很多优势，并且可以自带部分逻辑。我不使用txt格式。
        - langchain导入。langchain的prompt-template有太多优势。

    Args:
        prompt_template_path: prompt-template在本地保存的路径。

    Returns:
        可以直接进行format操作的prompt-template。
    """
    prompt_template = PromptTemplate.from_file(
        template_file=prompt_template_path,
        template_format='jinja2',  # 需要指定，否则解析方式不同。
        encoding='utf-8'  # 需要指定，否则解码中文有问题。
    )
    return prompt_template


# 旧的我自己构建的从指定路径加载prompt-template的方法。
# def load_prompt_template(prompt_template_path: str) -> str:
#     with open(prompt_template_path, 'r', encoding='utf-8') as f:
#         prompt_template = f.read()
#     return prompt_template


if __name__ == '__main__':
    pass
