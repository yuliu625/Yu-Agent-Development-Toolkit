"""
根据请求数量和时间的速率限制器。

文档来源:
    - openai: https://platform.openai.com/docs/guides/rate-limits
    - google: https://ai.google.dev/gemini-api/docs/rate-limits
    - anthropic: https://docs.anthropic.com/en/api/rate-limits
    - dashscope: https://help.aliyun.com/zh/model-studio/rate-limit
    - deepseek: https://api-docs.deepseek.com/zh-cn/quick_start/rate_limit

需要持续更新。
"""

from __future__ import annotations

from langchain_core.rate_limiters import InMemoryRateLimiter

from typing import TYPE_CHECKING, Literal
# if TYPE_CHECKING:


class RateLimiterFactory:
    """
    以strategy-pattern封装的速率限制器。
    """

    # ====主要方法。====
    @staticmethod
    def get_rate_limiter(
        model_client: Literal['openai', 'google', 'anthropic', 'dashscope', 'deepseek'],
        model_name: str,
        llm_number: int = 1,
    ) -> InMemoryRateLimiter:
        if model_client == 'openai':
            return RateLimiterFactory.get_openai_rate_limiter(model_name=model_name, llm_number=llm_number)
        elif model_client == 'google':
            return RateLimiterFactory.get_google_rate_limiter(model_name=model_name, llm_number=llm_number)
        elif model_client == 'anthropic':
            return RateLimiterFactory.get_anthropic_rate_limiter(model_name=model_name, llm_number=llm_number)
        elif model_client == 'dashscope':
            return RateLimiterFactory.get_dashscope_rate_limiter(model_name=model_name, llm_number=llm_number)
        elif model_client == 'deepseek':
            return RateLimiterFactory.get_deepseek_rate_limiter(model_name=model_name, llm_number=llm_number)

    @staticmethod
    def get_openai_rate_limiter(
        model_name: str,
        llm_number: int = 1,
    ) -> InMemoryRateLimiter:
        return InMemoryRateLimiter(
            requests_per_second=max(60/60 / llm_number, 1),
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )

    @staticmethod
    def get_google_rate_limiter(
        model_name: str,
        llm_number: int = 1,
    ) -> InMemoryRateLimiter:
        if model_name in (
            'gemini-2.5-pro',
        ):
            # tier 1
            return InMemoryRateLimiter(
                requests_per_second=max(150/60 / llm_number, 1),
                check_every_n_seconds=0.1,
                max_bucket_size=10,
            )
            # tier 2
            pass
        elif model_name in (
            'gemini-2.5-flash',
        ):
            # free tier
            # return InMemoryRateLimiter(
            #     requests_per_second=max(10/60 / llm_number, 1),
            #     check_every_n_seconds=0.1,
            #     max_bucket_size=10,
            # )
            # tier 1
            return InMemoryRateLimiter(
                requests_per_second=max(1000/60 / llm_number, 1),
                check_every_n_seconds=0.1,
                max_bucket_size=10,
            )
            # tier 2
            pass
        else:
            return InMemoryRateLimiter(
                requests_per_second=max(10/60 / llm_number, 1),
                check_every_n_seconds=0.1,
                max_bucket_size=10,
            )

    @staticmethod
    def get_anthropic_rate_limiter(
        model_name: str,
        llm_number: int = 1,
    ) -> InMemoryRateLimiter:
        return InMemoryRateLimiter(
            requests_per_second=max(60/60 / llm_number, 1),
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )

    @staticmethod
    def get_dashscope_rate_limiter(
        model_name: str,
        llm_number: int = 1,
    ) -> InMemoryRateLimiter:
        if model_name in (
            'qwen-max', 'qwen-max-latest', 'qwen-turbo', 'qwen-turbo-latest', 'qwen-long', 'qwen-long-latest',
            'qwen-vl-max', 'qwen-vl-max-latest', 'qwen-vl-plus', 'qwen-vl-plus-latest',
        ):
            return InMemoryRateLimiter(
                requests_per_second=max(1200/60 / llm_number, 1),
                check_every_n_seconds=0.1,
                max_bucket_size=10,
            )
        elif model_name in (
            'qwen-plus', 'qwen-plus-latest',
        ):
            return InMemoryRateLimiter(
                requests_per_second=max(15000/60 / llm_number, 1),
                check_every_n_seconds=0.1,
                max_bucket_size=10,
            )
        else:
            return InMemoryRateLimiter(
                requests_per_second=max(60/60 / llm_number, 1),
                check_every_n_seconds=0.1,
                max_bucket_size=10,
            )

    @staticmethod
    def get_deepseek_rate_limiter(
        model_name: str,
        llm_number: int = 1,
    ) -> InMemoryRateLimiter:
        """如果是deepseek本身云服务，并不限速。"""
        return InMemoryRateLimiter(
            requests_per_second=max(600/60 / llm_number, 1),
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )

