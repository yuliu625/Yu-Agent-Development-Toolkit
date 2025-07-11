"""
异步编程相关。

主要用于LLM development。
这只是一些改造工具，仅做参考，原始项目是可以一开始就写成异步的。
"""

from __future__ import annotations
import asyncio

from typing import Callable, Tuple, Any, Coroutine


async def async_wrap(sync_func: Callable, *args, **kwargs) -> Coroutine[Any, Any, Any]:
    """
    将同步程序包装为异步程序。

    Args:
        sync_func (Callable): 原本的同步程序。

    Returns:
        可以使用使用异步编程的协程。
    """
    return asyncio.to_thread(sync_func, *args, **kwargs)


async def run_parallel(async_func: Callable, arg_list: list[Tuple]) -> Tuple[Any]:
    """
    并行运行大量协程。

    如果是自构建的本身就是异步的函数，则不需要专门使用这个函数，而是在实现的时候就写为异步并发。

    Args:
        async_func (Callable): 目标要运行的异步程序。
        arg_list (list[Tuple(args, kwargs)]): 需要传输给目标异步函数的参数，包括args和kwargs。

    Returns:
        Tuple[result1, result2, ...]: 以原本调用顺序的结果。
    """
    tasks = [async_func(*args, **kwargs) for args, kwargs in arg_list]
    return await asyncio.gather(*tasks)

