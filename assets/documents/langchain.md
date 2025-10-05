### langchain
基于`langchain`的工具。整体而言，我很喜欢langchain中的`Runnable`和`Serializable`的设计。这样扎实的基础可以非常有助于高层级agent的各种构建。目前构建的工具有:
- **base_agent:** 通用任务agent。我早期构建的很棒的超类agent，具有agent开发必要的结构化输出和自我检验和重试机制。默认是LLM-based agent，记忆由外部实现进行灵活管理，例如由`langgraph`的state和具体节点逻辑控制。
- **model_factory:** 以工厂模式封装的LLM构建方法。统一创建和控制LLM的构建，支持常见科研任务需要的LLM。
- **prompt_management:** prompt管理工具。
  - **prompt_template_loader:** 封装了`langchain`中提供的prompt加载方法。约定基于`jinja2`定义和管理prompt-template。
  - **base_prompt_template_factory:** 获取各种prompt-template的方法。简化更换具体prompt-template的操作，使用常见科研任务的对比实验和消融实验。
  - **safe_format_message_prompt_template:** 安全format各种prompt-template的方法。根本上避免`f-string`和`jinja2`等模板语法的不严格变量检验。
- **model_building_tools:** 常见的对于LLM的设置和构建工具。
  - **rate_limiter_factory:** 持续更新的请求限速器。
  - **model_configs_builder:** 持续更新的LLM设置方法。
- **utils:** langchain相关的通用工具。
  - **merge_chunks:** 优雅的处理流式传输结果的方法。
  - **reasoning_content_processor:** 统一的对于Reasoning-Model生成内容的处理工具类。集中的处理和转换各种标准的reasoning_content的方法，用于持续对话和结果分析。
  - **message_copier:** 安全复制langchain中BaseMessage的方法。
- **embedding_model_factory:** 获取embedding-model的方法。

