"""
来自 llm_factory ，但是对 client 带有缓存。

对于创建 client 需要的资源很小，多个 model-client 很有利于并行加速和资源管理，因此这个 factory 很少使用。

Refactor:
    cache 应该由独立构建的 microservice 实现，不应该入侵 agent system 。
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

from typing import Annotated


class CachedLLMFactory:
    ...

