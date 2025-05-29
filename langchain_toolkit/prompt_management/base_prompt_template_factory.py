"""
通用的prompt-template-factory的基础实现。

需要:
    - prompt-template-loader。
"""

# 需要的该包中的其他工具。引入其他项目建议直接将2个文件都复制，再构建具体的prompt_template_factory，从而完全不修改这2个文件。
from .prompt_template_loader import PromptTemplateLoader

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain.prompts import PromptTemplate


class BasePromptTemplateFactory:
    """
    prompt-template-factory的基础方法。

    默认:
        - 使用jinja2文件管理prompt-template。不向下兼容。

    封装了:
        - 路径管理。不同的prompt文件树管理，默认使用2级文件夹层次化管理。
        - prompt加载。返回langchain可用的PromptTemplate。
    """
    def __init__(
        self,
        prompt_templates_dir: str | Path = None,
    ):
        """
        指定prompt-template文件存放的文件夹。

        如果不指定具体文件夹，默认会使用该文件所在的同一文件夹。

        Args:
            prompt_templates_dir (Union[str, Path, None]): 存放prompt-template的文件夹的路径。
        """
        if prompt_templates_dir is None:
            self.prompt_templates_dir = Path(__file__).parent
        else:
            self.prompt_templates_dir = Path(prompt_templates_dir)

    # ====主要方法。====
    def get_prompt_template(
        self,
        prompt_template_name: str,
    ) -> PromptTemplate:
        """
        主要方法。从文件加载prompt-template。

        使用我已经构建的加载工具。
        配套方法，封装原始工具为strategy-pattern。

        Args:
            prompt_template_name (str): prompt-template的名字。不含扩展名，默认指定为j2文件。

        Returns:
            PromptTemplate: langchain.prompts.PromptTemplate，接入langchain的runnable系统。
                system-prompt-template和message-prompt-template通用。
                仍需要进行的相关具体操作包括:
                    - partial
                    - invoke
                    - format
        """
        prompt_template_path = self.prompt_templates_dir / f"{prompt_template_name}.j2"
        prompt_template = PromptTemplateLoader.load_prompt_template_from_j2(prompt_template_path=prompt_template_path)
        return prompt_template

    def get_system_message(
        self,
    ) -> SystemMessage:
        ...

    def get_chat_prompt_template(
        self,
    ) -> ChatPromptTemplate:
        ...

    # ====控制方法。派生类仅调用一次。====
    def _set_sub_dir(
        self,
        sub_dir: str | Path
    ) -> None:
        """
        指定子文件夹。

        进行分类控制，代码和prompt-template文件分离。
        默认该方法由继承的派生类仅调用一次。

        Args:
            sub_dir (Union[str, Path]): 子文件夹的相对路径。

        Returns:
            None: 直接修改该类实例化对象的属性。
        """
        sub_dir = Path(sub_dir)
        self.prompt_templates_dir = self.prompt_templates_dir / sub_dir

