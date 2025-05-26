"""
来自llm_factory，但是对client带有缓存。

对于创建client需要的资源很小，多个model-client很有利于并行加速和资源管理，因此这个factory很少使用。
"""


from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv

from typing import Annotated


class CachedLLMFactory:
    ...

