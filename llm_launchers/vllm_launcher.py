"""
VLLM的启动器。

目的:
    - 封装shell输入: 使用subprocess封装用shell启动和配置vllm的输入。
    - 复用: 自动参数处理。
"""

from __future__ import annotations

import subprocess

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class VLLMLauncher:
    ...

