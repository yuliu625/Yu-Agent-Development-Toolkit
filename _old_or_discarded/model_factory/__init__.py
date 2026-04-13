"""
Factories that create LLM objects.

Notes:
    各种模型的构建应该以统一的方式进行，否则随着系统的扩大，中心的 LLM 构造的复杂度会持续提升。

    Refactor:
        - gateway: 以中间层隔离各种模型的区别，并且实现统一调度。
            差异不应该传导至 Agent System 层面。
        - microservice: 需要 cache 的地方应该分离，独立允许。
            推理不应该由 Agent System 控制。
"""

