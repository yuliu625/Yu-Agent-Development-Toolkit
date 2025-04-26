"""
异步编程相关。
主要用于LLM development。
这是一些改造工具，自己的代码可以一开始就写成异步的。
"""

import asyncio

from typing import Callable, List, Tuple, Any, Coroutine


async def async_wrap(sync_func: Callable, *args, **kwargs) -> Coroutine[Any, Any, Any]:
    """
    将同步程序包装为异步程序。
    Args:
        - sync_func: 原本的同步程序。
        - args: 位置参数。
        - kwargs: 关键字参数。
    Return:
        可以使用使用异步编程的协程。
    """
    return asyncio.to_thread(sync_func, *args, **kwargs)


async def run_parallel(async_func: Callable, arg_list: List[Tuple]) -> Tuple[Any]:
    """
    并行运行大量协程。
    Args:
        - async_func: 异步程序。
        - arg_list: List[Tuple(args, kwargs)]
    Return:
        Tuple[result1, result2, ...]，以原本调用顺序的结果。
    """
    tasks = [async_func(*args, **kwargs) for args, kwargs in arg_list]
    return await asyncio.gather(*tasks)


if __name__ == '__main__':
    pass
