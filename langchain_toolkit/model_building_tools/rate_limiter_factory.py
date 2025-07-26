"""
根据请求数量和时间的速率限制器。

文档来源:
    - openai: https://platform.openai.com/docs/guides/rate-limits https://platform.openai.com/docs/models
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

    参数:
        - model_client, model_name: 同llm-factory系列。
        - llm_number: 同时使用相关llm的数量。
            - 简单实现: 每个llm有独立的rate-limiter。
                更加严格，保证不会出错。
            - 改进: 不指定该参数。同一系列llm共享同一个rate-limiter，需要高层实现额外相关资源分配。
                更加高效灵活利用资源，尤其适用限制高的低速场景。
    """
    # ====主要方法。====
    @staticmethod
    def create_rate_limiter(
        model_client: Literal['openai', 'google', 'anthropic', 'dashscope', 'deepseek', 'local'],
        model_name: str,
        llm_number: int = 1,
    ) -> InMemoryRateLimiter:
        if model_client == 'openai':
            return RateLimiterFactory.create_openai_rate_limiter(model_name=model_name, llm_number=llm_number)
        elif model_client == 'google':
            return RateLimiterFactory.create_google_rate_limiter(model_name=model_name, llm_number=llm_number)
        elif model_client == 'anthropic':
            return RateLimiterFactory.create_anthropic_rate_limiter(model_name=model_name, llm_number=llm_number)
        elif model_client == 'dashscope':
            return RateLimiterFactory.create_dashscope_rate_limiter(model_name=model_name, llm_number=llm_number)
        elif model_client == 'deepseek':
            return RateLimiterFactory.create_deepseek_rate_limiter(model_name=model_name, llm_number=llm_number)
        elif model_client == 'local':
            return RateLimiterFactory.create_local_llm_rate_limiter(model_name=model_name, llm_number=llm_number)

    @staticmethod
    def create_openai_rate_limiter(
        model_name: str,
        llm_number: int = 1,
    ) -> InMemoryRateLimiter:
        """
        OpenAI对于不同付费等级有不同限速，需要具体选择。
        """
        # free tier
        # tier 1
        # tier 2
        return InMemoryRateLimiter(
            requests_per_second=60/60 / llm_number,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )

    @staticmethod
    def create_google_rate_limiter(
        model_name: str,
        llm_number: int = 1,
    ) -> InMemoryRateLimiter:
        """
        Google云服务对于不同付费等级有不同限速，需要具体选择。
        """
        if model_name in (
            'gemini-2.5-pro',
        ):
            # free tier
            # free tier can't use gemini-2.5-pro
            # tier 1
            return InMemoryRateLimiter(
                requests_per_second=150/60 / llm_number,
                check_every_n_seconds=0.1,
                max_bucket_size=10,
            )
            # tier 2
            # return InMemoryRateLimiter(
            #     requests_per_second=1000/60 / llm_number,
            #     check_every_n_seconds=0.1,
            #     max_bucket_size=10,
            # )
        elif model_name in (
            'gemini-2.5-flash',
        ):
            # free tier
            # return InMemoryRateLimiter(
            #     requests_per_second=10/60 / llm_number,
            #     check_every_n_seconds=0.1,
            #     max_bucket_size=10,
            # )
            # tier 1
            return InMemoryRateLimiter(
                requests_per_second=1000/60 / llm_number,
                check_every_n_seconds=0.1,
                max_bucket_size=10,
            )
            # tier 2
            # return InMemoryRateLimiter(
            #     requests_per_second=2000/60 / llm_number,
            #     check_every_n_seconds=0.1,
            #     max_bucket_size=10,
            # )
        else:
            return InMemoryRateLimiter(
                requests_per_second=10/60 / llm_number,
                check_every_n_seconds=0.1,
                max_bucket_size=10,
            )

    @staticmethod
    def create_anthropic_rate_limiter(
        model_name: str,
        llm_number: int = 1,
    ) -> InMemoryRateLimiter:
        return InMemoryRateLimiter(
            requests_per_second=60/60 / llm_number,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )

    @staticmethod
    def create_dashscope_rate_limiter(
        model_name: str,
        llm_number: int = 1,
    ) -> InMemoryRateLimiter:
        """
        阿里云模型有多种架构和多个checkpoint，根据具体情况选择和更新。
        """
        if model_name in (
            'qwen-max', 'qwen-max-latest',
            'qwen-turbo', 'qwen-turbo-latest',
            'qwen-long', 'qwen-long-latest',
            'qwen-vl-max', 'qwen-vl-max-latest',
            'qwen-vl-plus', 'qwen-vl-plus-latest',
        ):
            return InMemoryRateLimiter(
                requests_per_second=1200/60 / llm_number,
                check_every_n_seconds=0.1,
                max_bucket_size=10,
            )
        elif model_name in (
            'qwen-plus', 'qwen-plus-latest',
        ):
            return InMemoryRateLimiter(
                requests_per_second=15000/60 / llm_number,
                check_every_n_seconds=0.1,
                max_bucket_size=10,
            )
        else:
            return InMemoryRateLimiter(
                requests_per_second=60/60 / llm_number,
                check_every_n_seconds=0.1,
                max_bucket_size=10,
            )

    @staticmethod
    def create_deepseek_rate_limiter(
        model_name: str,
        llm_number: int = 1,
    ) -> InMemoryRateLimiter:
        """
        如果是deepseek本身云服务，并不限速。

        仅注意:
            - 不要到达请求发送机器的最大上限。
            - 不要因过长响应时间而判断为超时。
        """
        return InMemoryRateLimiter(
            requests_per_second=600/60 / llm_number,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )

    @staticmethod
    def create_local_llm_rate_limiter(
        model_name: str,
        llm_number: int = 1,
    ) -> InMemoryRateLimiter:
        """
        根据具体机器额外实现。
        """
        return InMemoryRateLimiter(
            requests_per_second=6000/60 / llm_number,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )

