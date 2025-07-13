"""
项目环境变量的处理工具。
"""

from __future__ import annotations

from dotenv import load_dotenv

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class DotEnvProcessor:
    @staticmethod
    def load_dotenv(
        dotenv_path: str = None,
    ) -> None:

        """
        指定dotenv_path仅为项目相关的情况，例如使用项目的API_KEY。
        这些初始化操作可以更大的项目中删除并自行管理环境变量的导入。

        Args:
            dotenv_path: (str), .env文件的路径。不指定会自行寻找。

        Returns:
            None: 加载.env文件中的变量至当前运行项目。
        """
        load_dotenv(dotenv_path=dotenv_path, override=True)

