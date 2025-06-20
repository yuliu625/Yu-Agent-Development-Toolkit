"""
进程管理器。独立持久运行通用工具。

预期场景:
    - 定时运行任务。
    - 运行可能会中断的实验，需要重启。
"""

from __future__ import annotations

import subprocess
import time
import sys
import datetime

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from subprocess import CompletedProcess


class ProcessManager:
    """
    进程运行管理工具。

    实现:
        - 暂时使用subprocess.run，为阻塞执行。
        - 后续使用subprocess.Popen，非阻塞执行并有更具体控制。
    """
    # ====基础方法。====
    @staticmethod
    def run_process(
        args: list[str],
        timeout: float | None = None,
    ) -> CompletedProcess[str] | None:
        try:
            result = subprocess.run(
                args=args,
                capture_output=True,
                text=True,
                encoding='utf-8',
                check=False,
                timeout=timeout,
            )
            return result
        except FileNotFoundError:
            print("error: file not found")
        except Exception as e:
            print(e)
        return None

    # ====主要方法。====
    @staticmethod
    def run_process_with_restart(
        args: list[str],
        timeout: float | None = None,
        max_retries: int = 10,
        retry_interval: int = 3,
    ):
        for i in range(max_retries):
            result = ProcessManager.run_process(
                args=args,
                timeout=timeout,
            )
            if result is None:
                return None
            if result.returncode == 0:
                return result
            else:
                time.sleep(retry_interval)
        return None

    # ====主要方法。====
    @staticmethod
    def run_processes_with_restart(
        args_list: list[list[str]],
        timeout: float | None = None,
        max_retries: int = 10,
        retry_interval: int = 3,
    ):
        for args in args_list:
            result = ProcessManager.run_process_with_restart(
                args=args,
                timeout=timeout,
                max_retries=max_retries,
                retry_interval=retry_interval,
            )

    @staticmethod
    def run_process_with_time_control(

    ):
        ...

    @staticmethod
    def batch_run_process(

    ):
        ...


if __name__ == '__main__':
    # 默认情况，这个文件可以直接作为根文件启动。
    process_args = [sys.executable, r"D:\document\code\environment\Yu-Agent-Development-Toolkit\agnostic_utils\t.py"]
    ProcessManager.run_process_with_restart(args=process_args)

