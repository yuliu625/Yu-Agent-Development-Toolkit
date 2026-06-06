### agnostic utils
一些和具体agent-framework无关的工具，但是是agent开发中必要的常使用的工具。目前构建的工具有:
- **json_output_extractor:** 从原始文本中提取符合json类型的结构化数据。我早期构建的非常棒的工具，考虑到各种结构化数据提取需求，完全封装了数据检验和修复方法。
- **json_input_processor:** 以标准的结构化数据格式输入的处理工具类。与`JsonOutputExtractor`一定程度互为相反方法，虽然这个方法很简单。
- **structured_data_extractor:** 进行结构化数据提取工具。`JsonOutputExtractor`的简化形式，为一定会有结构化数据输出的任务而设计，例如llm-as-a-labeler。
- **content_block_processor:** 针对`VLM`的内容块处理方法。为多模态大模型任务而使用的content构建和处理方法。
- **content_annotator:** 内容标记器。给一段文本标记tag。主要面临multi-agent场景标记不同agent的身份，以及RAG场景标记填充数据的metadata。

