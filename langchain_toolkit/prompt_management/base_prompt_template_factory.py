"""
通用的prompt-template-factory的基础实现。

需要:
    - PromptTemplateLoader。

约定:
    - 以前缀区分prompt-template。
"""

from __future__ import annotations

# 需要的该包中的其他工具。引入其他项目建议直接将2个文件都复制，再构建具体的prompt_template_factory，从而完全不修改这2个文件。
from .prompt_template_loader import PromptTemplateLoader

from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate


class BasePromptTemplateFactory:
    """
    prompt-template-factory的基础方法。

    这个实现用于agent设定和持续对话。可以使用PromptTemplateLoader构建其他的应用。

    约定:
        - 使用jinja2文件管理prompt-template。不向下兼容。
        - 以加载system-message-prompt-template获取chat-prompt-template。
        - 以加载human-message-prompt-template进行持续对话。

    预期使用:
        - system_message_prompt_template和human_message_prompt_template在同一文件夹，以名称前缀区分类别。
        - 所有的派生类，仅在构造方法中额外选择sub-dir。

    封装:
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
        指定的目的仅仅为从包外部获取prompt-template。

        Args:
            prompt_templates_dir (Union[str, Path, None]): 存放prompt-template的文件夹的路径。
        """
        if prompt_templates_dir is None:
            self.prompt_templates_dir = Path(__file__).parent
        else:
            self.prompt_templates_dir = Path(prompt_templates_dir)

    # ====主要方法。====
    def get_chat_prompt_template(
        self,
        system_message_prompt_template_name: str,
    ) -> ChatPromptTemplate:
        """
        可持续使用的chat-prompt-template。已给出system-message-prompt-template。

        如果system-message-prompt-template中含有参数，使用partial方法，再进行持续对话。

        注意:
            - 构建message-prompt-template时的命名，需要以 system_message_prompt_template_ 作为前缀。

        Args:
            system_message_prompt_template_name (str): strategy-patten封装对于message-prompt-template的获取。

        Returns:
            ChatPromptTemplate: 可持续使用的chat-prompt-template。
        """
        # 处理路径。
        system_message_prompt_template_path = (
            self.prompt_templates_dir
            # / 'system'  # 取消这行注释，修改和增加sub-dir。但默认不这样做。
            / f"system_message_prompt_template_{system_message_prompt_template_name}.j2"
        )
        # 加载。
        chat_prompt_template = PromptTemplateLoader.load_chat_prompt_template_from_j2(
            system_message_prompt_template_path=system_message_prompt_template_path
        )
        return chat_prompt_template

    # ====主要方法。====
    def safe_get_chat_prompt_template(
        self,
        system_message_prompt_template_name: str,
        system_message_prompt_template_format_kwargs: dict | None = None,
    ) -> ChatPromptTemplate:
        """
        可持续使用的chat-prompt-template。已给出system-message-prompt-template。

        如果system-message-prompt-template中的参数，已执行format方法。

        注意:
            - 构建message-prompt-template时的命名，需要以 system_message_prompt_template_ 作为前缀。

        Args:
            system_message_prompt_template_name (str): strategy-patten封装对于message-prompt-template的获取。
            system_message_prompt_template_format_kwargs (Union[dict, None]): 对

        Returns:
            ChatPromptTemplate: 可持续使用的chat-prompt-template。
        """
        # 处理路径。
        system_message_prompt_template_path = (
            self.prompt_templates_dir
            # / 'system'  # 取消这行注释，修改和增加sub-dir。但默认不这样做。
            / f"system_message_prompt_template_{system_message_prompt_template_name}.j2"
        )
        # 加载。
        chat_prompt_template = PromptTemplateLoader.safe_load_chat_prompt_template_from_j2(
            system_message_prompt_template_path=system_message_prompt_template_path,
            system_message_prompt_template_format_kwargs=system_message_prompt_template_format_kwargs,
        )
        return chat_prompt_template

    # ====主要方法。====
    def get_message_prompt_template(
        self,
        message_prompt_template_name: str,
    ) -> HumanMessagePromptTemplate:
        """
        加载持续对话的message-prompt-template。

        在agent-system中，给出指令的身份是human。

        注意:
            - 构建message-prompt-template时的命名，需要以 human_message_prompt_template_ 作为前缀。

        Args:
            message_prompt_template_name (str): strategy-patten封装对于message-prompt-template的获取。

        Returns:
            HumanMessagePromptTemplate: 用于和agent对话的message-prompt-template。
        """
        # 处理路径。
        human_message_prompt_template_path = (
            self.prompt_templates_dir
            # / 'message'  # 取消这行注释，修改和增加sub-dir。但默认不这样做。
            / f"human_message_prompt_template_{message_prompt_template_name}.j2"
        )
        # 加载。
        message_prompt_template = PromptTemplateLoader.load_human_message_prompt_template_from_j2(
            human_message_prompt_template_path=human_message_prompt_template_path,
        )
        return message_prompt_template

    # ====冗余方法。====
    def get_prompt_template(
        self,
        prompt_template_name: str,
    ) -> PromptTemplate:
        """
        冗余方法。从文件加载prompt-template。

        使用我已经构建的加载工具。配套方法，封装原始工具为strategy-pattern。
        这个方法会直接操作字符串，少数情况可能使用。

        Args:
            prompt_template_name (str): prompt-template的名字。不含扩展名，约定指定为j2文件。

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

    # ====控制方法。派生类最多在构造方法中调用一次。====
    def set_sub_dir(
        self,
        sub_dir: str | Path
    ) -> None:
        """
        指定子文件夹。

        进行分类控制，代码和prompt-template文件分离。

        使用情况:
            - 多角色。
            - ablation-study。

        注意:
            - 约定该方法由继承的派生类仅调用一次。
            - system_message_prompt_template和human_message_prompt_template的控制由具体方法实现，而不是对象所处的路径。

        Args:
            sub_dir (Union[str, Path]): 子文件夹的相对路径。

        Returns:
            None: 直接修改该类实例化对象的属性。
        """
        sub_dir = Path(sub_dir)
        self.prompt_templates_dir = self.prompt_templates_dir / sub_dir
        print(f"set prompt-template-dir to {self.prompt_templates_dir}")

