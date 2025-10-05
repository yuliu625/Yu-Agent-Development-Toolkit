# Agent Development Toolkit

## 概述
这是一个致力于 LLM-based Agent 开发的综合、工程化工具集。

本仓库的核心目标是提供一套经过实战验证、高复用、且健壮的工具和组件，专门用于构建和管理复杂的 AI Agent 系统。我将工程化的重点放在 稳定性、流程可控性 和 系统可扩展性 上，旨在快速进行原型开发和生产级 Agent 应用过渡。

### 核心技术栈
我选择并深入定制了 LangChain 和 LangGraph 体系，作为构建复杂 Agent 系统的基础:
- LangChain: 充分利用其 Runnable 和 Serializable 抽象，为构建高层级、可组合的 Agent 组件提供了扎实的基础。
- LangGraph: 核心优势在于其状态机模型。这使得我能够对 Agent 的决策流程和循环进行精确控制和灵活定义，特别适合需要复杂规划、人机交互或工具使用的场景。

*背景说明: 我曾尝试基于 Autogen 和 Llama-Index 进行 Agent 研发，但实践证明，在面对需要细粒度流程控制和高稳定性要求的复杂 Agent 系统时，LangChain + LangGraph 体系在代码工程化层面更具优势。*


## 仓库结构与核心功能模块
本仓库的工具集被划分为几个功能清晰的模块，以应对 Agent 开发中的常见挑战:
### 通用核心工具
提供与具体 Agent 框架无关，但在任何 LLM/Agent 项目中都必不可少的基础工具，专注于输入/输出的可靠性。
- **结构化数据处理**: 包含用*安全提取 JSON、处理结构化输入、以及简化的结构化数据提取工具。重点解决了 LLM 输出格式不稳定和校验困难的问题。
- **内容与上下文处理**: 提供了 VLM 的内容块处理（多模态输入构建）和内容标记器用于多 Agent 身份或 RAG 元数据标记。

### LangChain 扩展与工具集
基于 LangChain 体系，提供了一系列用于增强 LLM Agent 稳定性和工程效率的组件。
- **`base_agent`**: 通用 Agent 的超类，内置结构化输出、自我检验和重试机制。
- **模型构建与管理**: 统一的 LLM 工厂、Embedding Model工厂以及配置构建器，简化了模型切换和配置管理。
- **Prompt 工程**: 封装了基于 `jinja2` 的 Prompt 加载和管理，并提供了安全格式化方法，以避免模板变量错误但不进行任何报错等风险。
- **实用工具**: 包含用于优雅处理流式传输、Reasoning 内容统一处理和消息安全复制等工具。

### Langgraph 流程控制与图构建
专注于利用 LangGraph 的状态机特性，实现对 Agent 工作流的精确控制。
- **`graph_builder`**: 定义了计算图的构建标准，帮助规范化定义复杂的 Agent 循环和逻辑。
- **实用工具**: 辅助管理 LangGraph 相关状态和节点。

### LLM Launchers: 统一 LLM 推理服务
一个专门的模块，用于统一管理和简化各种 LLM 推理服务（包括本地和远程 API）的使用和部署。旨在抽象化繁琐的服务配置细节。

*其他框架: 你可以在`_other_agent_frameworks`查看我早期基于Autogen和Llama-Index构建的部分工具。*


## 更多我的 Agent 与深度学习项目
欢迎查看我的其他专注于特定领域或研究的仓库，以了解更多工程和研究工作，这些项目与 Agent Development Toolkit 共同构成了我的技术生态。
### 通用工具集
- [Deep-Learning-Toolkit](https://github.com/yuliu625/Yu-Deep-Learning-Toolkit): 一个用于深度学习任务的通用工具集。
- [Data-Science-Toolkit](https://github.com/yuliu625/Yu-Data-Science-Toolkit): 我为数据科学任务构建的工具集。
### 现有 Agent 研究工作
- [Simulate-the-Prisoners-Dilemma-with-Agents](https://github.com/yuliu625/Simulate-the-Prisoners-Dilemma-with-Agents): 我基于 Autogen 的早期尝试，研究 LLM Agent 在囚徒困境等简单博弈场景下的决策行为。
- [World-of-Six](https://github.com/yuliu625/World-of-Six): 我对 Agent 在具有网络效应环境下的决策行为研究。(论文已被 SWAIB[2025] 接收)
### 正在进行中的项目(未来开源)
- 正在进行一项关于 LLM-based Agent 在具有网络效应的环境下的预期行为的研究（相关代码将在期刊发表后开源）。
- 正在研究一个通过构建 Multi-agent System 来分析财务报告的文档智能项目。

