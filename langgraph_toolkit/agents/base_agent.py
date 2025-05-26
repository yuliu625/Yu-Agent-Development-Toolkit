"""
基础的agent。
"""

from langchain_core.runnables import Runnable


class BaseAgent:
    def __init__(
        self,
        runnable: Runnable,
    ):
        self.runnable = runnable

    async def __call__(self, *args, **kwargs):
        ...

