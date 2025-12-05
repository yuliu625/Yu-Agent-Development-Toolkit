"""
Document source: https://docs.vllm.ai/en/stable/cli/serve/

VLLM的启动器。

VLLM的设计使得在部署推理服务时，使用CLI是更好的选择。然而大量基于 shell 的输入很不方便，因此我进行了封装。
这个实现是基于 VLLM V1，主要场景是本地推理服务。

目的:
    - 封装shell输入: 使用subprocess封装用shell启动和配置vllm的输入。
    - 复用: 自动参数处理。
    - 隔离环境: 独立推理环境，减少开发环境依赖管理复杂度。
"""

from __future__ import annotations

import subprocess

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from subprocess import CompletedProcess


class VLLMLauncher:
    @staticmethod
    def start_vllm(
        mode: Literal['serve', 'chat'],  # 目前需求是serve和chat，后续添加其他方法。
        model: str,
        host: str | None = None,
        port: int | None = None,
        lora_modules: dict[str, str] | None = None,
    ) -> CompletedProcess[str]:
        """
        阻塞启动vllm的方法。

        Args:
            mode (Literal['serve', 'chat']): 启动 vllm 的方式，根据文档进行的设置。
            model (str): 模型的路径。这个工具类主要可以简化和灵活设置的地方。
            host (str, optional): url的host部分。我默认为local host。
            port (int, optional): url的端口号。这会在同时运行多个模型时很有用。
            lora_modules (dict[str, str], optional): 加载adapter的具体设置。

        Returns:
            None: 运行vllm服务。这个实现是阻塞的。
        """
        command = [
            'vllm',  # V1版本中，不再用"python -m vllm.entrypoints.api_server"启动。
        ]
        # set the mode
        if mode == 'serve':
            command.extend(['serve'])
        elif mode == 'chat':
            command.extend(['chat'])
        # set the model path, which is necessary
        command.extend(['--model', model])
        # conditional add other args
        if host:
            command.extend(['--host', host])
        if port:
            command.extend(['--port', str(port)])
        if lora_modules:
            command.extend(['--lora-modules', lora_modules])
        # launch vllm server
        try:
            result = subprocess.run(
                args=command,
                text=True,
            )
            return result
        except Exception as e:
            print(e)

    @staticmethod
    def stop_vllm() -> None:
        """
        停止VLLM服务的运行。

        如果以subprocess.Popen非阻塞方式运行，需要实现停止的方法。
        """
        raise NotImplementedError


if __name__ == '__main__':
    # 一次设置以下参数即可。如果有多个模型需要部署，简单的，复制和配置该文件多次。
    VLLMLauncher.start_vllm(
        mode='chat',
        model=r'',
        host='127.0.0.1',
        port=8000,
        lora_modules=None,
    )

