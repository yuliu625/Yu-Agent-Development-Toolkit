"""
llm构建时model_configs这个参数的构造方法。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from langchain_core.rate_limiters import BaseRateLimiter


class ModelConfigsBuilder:
    """
    构建 model_configs 这个参数的方法。model_configs会用于构建llm。

    函数式编程，无后效性。
    """
    @staticmethod
    def build_model_configs(
        model_configs: dict,
        max_retries: int | None,
        model_client: Literal['openai', 'google', 'anthropic', 'dashscope', 'deepseek', 'ollama'],
        rate_limiter: BaseRateLimiter | None = None,
    ) -> dict:
        model_configs = ModelConfigsBuilder.add_max_retries(
            model_configs=model_configs,
            max_retries=max_retries,
        )
        model_configs = ModelConfigsBuilder.add_reasoning_kwargs(
            model_configs=model_configs,
            model_client=model_client,
        )
        model_configs = ModelConfigsBuilder.add_rate_limiter(
            model_configs=model_configs,
            rate_limiter=rate_limiter,
        )
        return model_configs

    # ====工具方法。====
    @staticmethod
    def add_rate_limiter(
        model_configs: dict,
        rate_limiter: BaseRateLimiter | None = None,
    ) -> dict:
        """
        添加限速器，设置模型的请求速度。

        可以不传入限速器，以实现不限速。
        约定的实现是添加一个限速器，改进的做法是限速器是共用的。

        Args:
            model_configs (dict): 模型相关的设置参数。
            rate_limiter (BaseRateLimiter, optional): 限速器。可以为None，代表不进行限速。

        Returns:
            dict: 添加 rate_limiter 之后的dict。和传入的dict独立。
        """
        model_configs_ = model_configs.copy()
        model_configs_['rate_limiter'] = rate_limiter
        return model_configs_

    # ====工具方法。====
    @staticmethod
    def add_max_retries(
        model_configs: dict,
        max_retries: int | None = None,
    ) -> dict:
        """
        设置模型可以进行重试。

        在默认情况下，让每个模型可以进行10次重试。
        该方法一定会配置这个max_retries以实现更健壮的请求。

        Args:
            model_configs (dict): 模型相关的设置参数。
            max_retries (int, optional): 最大重试次数。配置之后使得LLM可以进行retry。

        Returns:
            dict: 添加 max_retries 之后的dict。和传入的dict独立。
        """
        model_configs_ = model_configs.copy()
        model_configs_['max_retries'] = max_retries or 10
        return model_configs_

    # ====工具方法。====
    @staticmethod
    def add_reasoning_kwargs(
        model_configs: dict,
        model_client: Literal['openai', 'google', 'anthropic', 'dashscope', 'deepseek', 'ollama'],
    ) -> dict:
        """
        添加相关参数使得模型可以进行reasoning方式的响应。

        可以提前设置相关参数，不会覆盖已传入的参数。

        因为每个client的设置方式不同，根据具体情况进行添加设置。

        Args:
            model_configs (dict): 传递给langchain的BaseChatModel的设置参数。
            model_client (Literal['openai', 'google', 'anthropic', 'dashscope', 'deepseek', 'ollama']): 模型的client。

        Returns:
            dict: 可以进行reasoning的dict设置。
        """
        model_configs_ = model_configs.copy()
        # 根据每个client的情况添加参数。
        if model_client == 'openai':
            raise NotImplementedError
        elif model_client == 'google':
            raise NotImplementedError
        elif model_client == 'anthropic':
            raise NotImplementedError
        elif model_client == 'dashscope':
            model_configs_.setdefault('streaming', True)  # qwen系列思考模式只能流式传输。
            model_configs_['model_kwargs'].setdefault('enable_thinking', True)  # 具体需要设置的参数，开始思考模式。
            return model_configs_
        elif model_client == 'deepseek':
            # deepseek的reasoning模式不可设置，只有一种模式。
            return model_configs_
        elif model_client == 'ollama':
            raise NotImplementedError

