"""
Sources:
    https://github.com/yuliu625/Yu-Agent-Development-Toolkit/blob/main/src/langchain_toolkit/prompt_management/prompt_template_loader.py

References:
    None

Synopsis:
    从指定路径加载 prompt-template 的方法。

Notes:
    我实际工程的规范:
        - 从文件导入。几乎不硬编码 prompt-template 。因为:
            - 直接写字符串面临换行等问题，写起来很丑。
            - 方便进行版本控制。
        - jinja2 格式。不使用 txt 。因为:
            - json 友好。不用为 json 数据写额外的保护操作。例如，使用字符串拼接避免 format 操作。
            - 可以写逻辑。简化 format 操作，完全不使用拼接。不用额外构建 prompt_builder 。
            - 可以写注释。维护很方便。
        - PromptTemplate 和 MessagePromptTemplate 对象。不使用 f-string 。因为:
            - 更好的控制。例如 partial 。
        - 以 _prompt_template 命名结尾。区分 prompt 和 prompt-template 。

    实现细节:
        - message-prompt-template 的加载:
            prompt_template 加载直接使用 langchain 提供的 from_file 方法是没问题的，
            但是 message_prompt_template 的 from_template_file 方法有问题，具体是:
                - 强制要求 input_variables ，但是签名和 from_template 中不一样。不清楚这样设计的原因。
                - 无法指定 utf-8 编码，中文文件会出现问题。
            因此，message_prompt_template 为我使用 pathlib 修改的方法。不使用 from_template_file ，而是封装了 from_template 。
"""

from __future__ import annotations
from loguru import logger

from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from pathlib import Path

from typing import TYPE_CHECKING, Annotated, Literal
if TYPE_CHECKING:
    from langchain_core.prompts.chat import BaseMessagePromptTemplate


class PromptTemplateLoader:
    """
    从文件系统读取 prompt-template 的工具。

    可以在外层构建一个 prompt-template-factory ，实现完全用 strategy-pattern 控制 prompt-template 。

    主要方法:
        - load_prompt_template_from_j2: 最基础的加载 prompt-template 的方法，可以处理所有的字符串文本。
        - load_message_prompt_template_from_j2: 最常用的加载 message-prompt-template 的方法，用于构建 message-list 。
        - load_chat_prompt_template_from_j2: 预构建部分 llm-chain ，自带 system-prompt 。
    """

    # ====主要方法。====
    @staticmethod
    def load_prompt_template_from_j2(
        prompt_template_path: Annotated[str | Path, 'prompt-template所在的路径'],
    ) -> PromptTemplate:
        """
        从文件路径加载 prompt-template 的方法。

        Args:
            prompt_template_path (Union[str, Path]): prompt-template 在本地保存的路径。

        Returns:
            PromptTemplate: 可以进行 langchain 中相关操作的 prompt-template 。
        """
        prompt_template = PromptTemplate.from_file(
            template_file=prompt_template_path,
            template_format='jinja2',  # 需要指定，否则解析方式为 f-string 。
            encoding='utf-8',  # 需要指定，否则解码中文有问题。
        )
        return prompt_template

    # ====主要方法。====
    @staticmethod
    def load_message_prompt_template_from_j2(
        message_prompt_template_path: Annotated[str | Path, 'message-prompt-template所在的路径'],
        message_type: Literal['system', 'human', 'ai'],
    ) -> BaseMessagePromptTemplate:
        """
        从文件路径加载 message-prompt-template 的方法。

        用 strategy-pattern 进行封装，但是可以直接使用具体的工具类方法加载。

        Args:
            message_prompt_template_path (Union[str, Path]): message-prompt-template 在本地保存的路径。
            message_type (Literal['system', 'human', 'ai']): 加载的 message-prompt-template 的类型。

        Returns:
            BaseMessagePromptTemplate: 具体的 message-prompt-template 。
                实际上是 Union[SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate]。
                langchain_core.prompts.chat 中的源码实际是 _StringImageMessagePromptTemplate 。
        """
        if message_type == 'system':
            return PromptTemplateLoader.load_system_message_prompt_template_from_j2(
                system_message_prompt_template_path=message_prompt_template_path,
            )
        elif message_type == 'human':
            return PromptTemplateLoader.load_human_message_prompt_template_from_j2(
                human_message_prompt_template_path=message_prompt_template_path,
            )
        elif message_type == 'ai':
            return PromptTemplateLoader.load_ai_message_prompt_template_from_j2(
                ai_message_prompt_template_path=message_prompt_template_path,
            )

    # ====主要方法。====
    @staticmethod
    def load_chat_prompt_template_from_j2(
        system_message_prompt_template_path: str | Path,
        message_place_holder_key: str = 'chat_history',
    ) -> ChatPromptTemplate:
        """
        加载预构建的 llm-chain 的 system-prompt 的部分。

        这个方法的目的是分离 system-prompt-message 和一般消息的部分。

        Args:
            system_message_prompt_template_path (Union[str, Path]): system-message-prompt-template 的存储路径。
            message_place_holder_key (str, optional): 后续信息占位符 key 的命名。默认为 'chat_history' 。
                参与构建 llm-chain 后，后续 invoke 仅传入 messages 即可。
                对于 MessagesPlaceholder 可以进一步指定，但是因为并不常用，该方法并没有实现。

        Returns:
            ChatPromptTemplate: 可使用 invoke 传递 chat-history 的 chat-prompt-template 。
        """
        system_message_prompt_template = PromptTemplateLoader.load_system_message_prompt_template_from_j2(
            system_message_prompt_template_path=system_message_prompt_template_path,
        )
        chat_prompt_template = ChatPromptTemplate.from_messages(
            messages=[
                system_message_prompt_template,
                MessagesPlaceholder(message_place_holder_key),
            ],
            template_format='jinja2',
        )
        return chat_prompt_template

    # ====主要方法。====
    @staticmethod
    def safe_load_chat_prompt_template_from_j2(
        system_message_prompt_template_path: str | Path,
        system_message_prompt_template_format_kwargs: dict,
        message_place_holder_key: str = 'chat_history',
    ) -> ChatPromptTemplate:
        """
        加载预构建的 llm-chain 的 system-prompt 的部分。

        这个方法的目的是分离 system-prompt-message 和一般消息的部分。
        这个方法与 load_chat_prompt_template_from_j2 的区别是:
            - 在加载 system_message_prompt_template 后就执行 format 操作，从而避免意外参数导致的错误。
                但是，会失去一些灵活性，以及自构建 message-prompt-template 可以避免这些情况。

        Args:
            system_message_prompt_template_path (Union[str, Path]): system-message-prompt-template 的存储路径。
            system_message_prompt_template_format_kwargs (dict): 对 system_message_prompt_template 指定 format 操作的 kwargs 。
            message_place_holder_key (str, optional): 后续信息占位符 key 的命名。默认为 'chat_history' 。
                参与构建 llm-chain 后，后续 invoke 仅传入 messages 即可。
                对于 MessagesPlaceholder 可以进一步指定，但是因为并不常用，该方法并没有实现。

        Returns:
            ChatPromptTemplate: 可使用 invoke 传递 chat-history 的 chat-prompt-template 。
        """
        system_message_prompt_template = PromptTemplateLoader.load_system_message_prompt_template_from_j2(
            system_message_prompt_template_path=system_message_prompt_template_path,
        )
        # system_message_prompt_template_format_kwargs 为 None 时的兼容性处理。
        if system_message_prompt_template_format_kwargs:
            system_message = system_message_prompt_template.format(**system_message_prompt_template_format_kwargs)
        else:
            system_message = system_message_prompt_template.format()
        chat_prompt_template = ChatPromptTemplate.from_messages(
            messages=[
                system_message,
                MessagesPlaceholder(message_place_holder_key),
            ],
            template_format='jinja2',
        )
        return chat_prompt_template

    # ====常用方法。====
    @staticmethod
    def load_system_message_prompt_template_from_j2(
        system_message_prompt_template_path: Annotated[str | Path, 'message-prompt-template所在的路径'],
    ) -> SystemMessagePromptTemplate:
        template = Path(system_message_prompt_template_path).read_text(encoding='utf-8')
        system_message_prompt_template = SystemMessagePromptTemplate.from_template(
            template=template,
            template_format='jinja2',
        )
        return system_message_prompt_template

    # ====常用方法。====
    @staticmethod
    def load_human_message_prompt_template_from_j2(
        human_message_prompt_template_path: Annotated[str | Path, 'message-prompt-template所在的路径'],
    ) -> HumanMessagePromptTemplate:
        template = Path(human_message_prompt_template_path).read_text(encoding='utf-8')
        human_message_prompt_template = HumanMessagePromptTemplate.from_template(
            template=template,
            template_format='jinja2',
        )
        return human_message_prompt_template

    # ====为了完整性保留的方法。====
    @staticmethod
    def load_ai_message_prompt_template_from_j2(
        ai_message_prompt_template_path: Annotated[str | Path, 'message-prompt-template所在的路径'],
    ) -> AIMessagePromptTemplate:
        template = Path(ai_message_prompt_template_path).read_text(encoding='utf-8')
        ai_message_prompt_template = AIMessagePromptTemplate.from_template(
            template=template,
            template_format='jinja2',
        )
        return ai_message_prompt_template

    # ====已弃用。旧的从指定路径加载 prompt-template 的方法。====
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
            str: 读取的str文本。
        """
        file_path = Path(file_path)
        text = file_path.read_text(encoding='utf-8')
        return text

