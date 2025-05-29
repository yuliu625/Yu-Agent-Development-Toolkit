"""
基础的agent。
"""

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.runnables import Runnable


class BaseAgent:
    def __init__(
        self,
        llm_chain: Runnable,
    ):
        self.llm_chain = llm_chain

    async def __call__(self, *args, **kwargs):
        ...

