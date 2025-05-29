"""
以factory-pattern构建的LLM获取方法。

通用的factory，宽松的strategy-pattern。可以进一步指定进行二次改写。

这package包括:
    - llm_factory: 最主要的方法，封装了OpenAI-API的LLM。
    - local_llm_factory: 本地模型的使用方法，需要进行具体改写和指定。
"""

