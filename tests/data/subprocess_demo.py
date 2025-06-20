"""
模拟有可能会失败的子程序。
"""

from __future__ import annotations

import random
import sys
import time
import datetime

print(f"[{datetime.datetime.now()}] 正在运行模拟实验...")
run_duration = random.randint(1, 3)  # 模拟实验运行时间 1-3 秒
print(f"模拟实验将运行约 {run_duration} 秒。")
time.sleep(run_duration)

# 模拟因网络问题导致的失败（例如，随机失败）
if random.random() < 0.8:  # 80% 的概率失败
    print(f"[{datetime.datetime.now()}] 模拟网络问题，实验失败。")
    sys.exit(1)  # 非零退出码表示失败
else:
    print(f"[{datetime.datetime.now()}] 实验成功完成。")
    sys.exit(0)  # 零退出码表示成功
