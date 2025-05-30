"""
从指定路径加载prompt-template的方法。

我实际工程的规范:
    - 从文件导入。几乎不硬编码prompt-template。因为:
        - 直接写字符串面临换行等问题，写起来很丑。
        - 方便进行版本控制。
    - jinja2格式。不使用txt。因为:
        - json友好。不用为json数据写额外的保护操作。例如，使用字符串拼接避免 format 操作。
        - 可以写逻辑。简化format操作，完全不使用拼接。不用额外构建 prompt_builder 。
        - 可以写注释。维护很方便。
    - PromptTemplate和MessagePromptTemplate对象。不使用f-string。因为:
        - 更好的控制。例如partial。
    - 以_prompt_template命名结尾。区分prompt和prompt-template。

实现细节:
    - message-prompt-template的加载:
        prompt_template加载直接使用langchain提供的from_file方法是没问题的，
        但是message_prompt_template的from_template_file方法有问题，具体是:
            - 强制要求input_variables，但是签名和from_template中不一样。不清楚这样设计的原因。
            - 无法指定utf-8编码，中文文件会出现问题。
        因此，message_prompt_template为我使用pathlib修改的方法。不使用from_template_file，而是封装了from_template。
"""

from __future__ import annotations

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
    从文件系统读取prompt-template的工具。

    可以在外层构建一个prompt-template-factory，实现完全用strategy-pattern控制prompt-template。

    主要方法:
        - load_prompt_template_from_j2: 最基础的加载prompt-template的方法，可以处理所有的字符串文本。
        - load_message_prompt_template_from_j2: 最常用的加载message-prompt-template的方法，用于构建message-list。
        - load_chat_prompt_template_from_j2: 预构建部分llm-chain，自带system-prompt。
    """

    # ====主要方法。====
    @staticmethod
    def load_prompt_template_from_j2(
        prompt_template_path: Annotated[str | Path, 'prompt-template所在的路径'],
    ) -> PromptTemplate:
        """
        从文件路径加载prompt-template的方法。

        Args:
            prompt_template_path (Union[str, Path]): prompt-template在本地保存的路径。

        Returns:
            PromptTemplate: 可以进行langchain中相关操作的prompt-template。
        """
        prompt_template = PromptTemplate.from_file(
            template_file=prompt_template_path,
            template_format='jinja2',  # 需要指定，否则解析方式为f-string。
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
        从文件路径加载message-prompt-template的方法。

        用strategy-pattern进行封装，但是可以直接使用具体的工具类方法加载。

        Args:
            message_prompt_template_path (Union[str, Path]): message-prompt-template在本地保存的路径。
            message_type (Literal['system', 'human', 'ai']): 加载的message-prompt-template的类型。

        Returns:
            BaseMessagePromptTemplate: 具体的message-prompt-template。
                实际上是Union[SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate]。
                langchain_core.prompts.chat中的源码实际是 _StringImageMessagePromptTemplate 。
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
        加载预构建的llm-chain的system-prompt的部分。

        这个方法的目的是分离system-prompt-message和一般消息的部分。

        Args:
            system_message_prompt_template_path (Union[str, Path]): system-message-prompt-template的存储路径。
            message_place_holder_key (str, optional): 后续信息占位符key的命名。默认为'chat_history'。
                参与构建llm-chain后，后续invoke仅传入messages即可。
                对于MessagesPlaceholder可以进一步指定，但是因为并不常用，该方法并没有实现。

        Returns:
            ChatPromptTemplate: 可使用invoke传递chat-history的chat-prompt-template。
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
        加载预构建的llm-chain的system-prompt的部分。

        这个方法的目的是分离system-prompt-message和一般消息的部分。
        这个方法与load_chat_prompt_template_from_j2的区别是:
            - 在加载system_message_prompt_template后就执行format操作，从而避免意外参数导致的错误。
                但是，会失去一些灵活性，以及自构建message-prompt-template可以避免这些情况。

        Args:
            system_message_prompt_template_path (Union[str, Path]): system-message-prompt-template的存储路径。
            system_message_prompt_template_format_kwargs (dict): 对system_message_prompt_template指定format操作的kwargs。
            message_place_holder_key (str, optional): 后续信息占位符key的命名。默认为'chat_history'。
                参与构建llm-chain后，后续invoke仅传入messages即可。
                对于MessagesPlaceholder可以进一步指定，但是因为并不常用，该方法并没有实现。

        Returns:
            ChatPromptTemplate: 可使用invoke传递chat-history的chat-prompt-template。
        """
        system_message_prompt_template = PromptTemplateLoader.load_system_message_prompt_template_from_j2(
            system_message_prompt_template_path=system_message_prompt_template_path,
        )
        system_message = system_message_prompt_template.format(**system_message_prompt_template_format_kwargs)
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

    # ====已弃用。旧的从指定路径加载prompt-template的方法。====
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

