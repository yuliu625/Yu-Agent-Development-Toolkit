# Agent Development Toolkit

## 📖 Overview

This is a comprehensive, engineering-oriented toolkit dedicated to the development of LLM-based Agents.

The core objective of this repository is to provide a set of battle-tested, highly reusable, and robust tools and components specifically designed for building and managing complex AI Agent systems. I place a strong emphasis on **stability, process controllability, and system scalability**, aiming to facilitate rapid prototyping and the transition to production-grade Agent applications.

### 🗂️ Core Technology Stack

I have chosen and deeply customized the LangChain and LangGraph ecosystems as the foundation for building complex Agent systems:

- **LangChain**: Fully leveraging its `Runnable` and `Serializable` abstractions to provide a solid foundation for building high-level, composable Agent components.
- **LangGraph**: The core advantage lies in its state machine model. This allows for precise control and flexible definition of Agent decision-making flows and loops, making it particularly suitable for scenarios requiring complex planning, human-in-the-loop interactions, or extensive tool usage.

*Background Note: I previously explored Agent R&D based on AutoGen and Llama-Index. However, practice has shown that for complex Agent systems requiring fine-grained process control and high stability, the LangChain + LangGraph ecosystem offers superior advantages at the code engineering level.*

## 🚀 Repository Status

> To achieve ultimate engineering decoupling, I have optimized the Agent system by "separating logic from the underlying infrastructure." This repository has evolved into a pure toolkit for Agent decision-making and process control, no longer containing implementations for underlying inference or basic retrieval.

- **Underlying Inference & Deployment**: For content regarding LLM inference optimization, model serving, and unified API encapsulation, please refer to [AI-Inference-Stack](https://github.com/yuliu625/Yu-AI-Inference-Stack).
- **Knowledge Retrieval & RAG**: For specific implementations of vector databases, document splitting, and retrieval augmentation, please refer to [RAG-Toolkit](https://github.com/yuliu625/Yu-RAG-Toolkit).

## 🏗️ Repository Structure & Core Modules

The toolkit is divided into several clearly defined modules to address common challenges in Agent development:

### General Core Tools

Provides fundamental tools independent of specific Agent frameworks but essential for any LLM/Agent project, focusing on the reliability of inputs and outputs.

- **Structured Data Processing**: Includes utilities for safely extracting JSON, handling structured inputs, and simplified structured data extraction. It focuses on solving problems related to unstable LLM output formats and validation difficulties.
- **Content & Context Handling**: Provides VLM content block processing (for multimodal input construction) and content taggers for multi-agent identity or RAG metadata marking.

### LangChain Enhancement Components

Based on the LangChain ecosystem, these components enhance the stability and engineering efficiency of LLM Agents.

- **`base_agent`**: A superclass for general-purpose Agents with built-in structured output, self-verification, and retry mechanisms.
- **Model Construction & Management**: Unified LLM factories, Embedding Model factories, and configuration builders to simplify model switching and configuration management.
- **Prompt Engineering**: Encapsulated Prompt loading and management based on `jinja2`, providing safe formatting methods to avoid risks such as silent errors in template variables.
- **Utility Tools**: Includes tools for gracefully handling streaming, unified processing of "Reasoning" content, and secure message copying.

### LangGraph Process Control & Graph Construction
Focuses on leveraging LangGraph's state machine characteristics to achieve precise control over Agent workflows.
- **`graph_builder`**: Defines standards for building computational graphs, helping to standardize the definition of complex Agent loops and logic.
- **Utility Tools**: Helpers for managing LangGraph-related states and nodes.

### OpenAI Forge
A lightweight wrapper for the native OpenAI SDK, suitable for minimalist scenarios that do not require complex frameworks.

*Other Frameworks: You can find earlier tools built on AutoGen and Llama-Index in the `_other_agent_frameworks` directory.*

## 🌐 Personal Technical Ecosystem
Explore my other repositories focused on specific domains or research to learn more about my engineering and research work. These projects, together with the Agent Development Toolkit, constitute my technical ecosystem.

### General Toolkits
- **[AI-Inference-Stack](https://github.com/yuliu625/Yu-AI-Inference-Stack)**: Tools for inference serving.
- **[RAG-Toolkit](https://github.com/yuliu625/Yu-RAG-Toolkit)**: Core retrieval-augmented generation toolkit.
- **[Agent-Development-Toolkit](https://github.com/yuliu625/Yu-Agent-Development-Toolkit)**: Focused on Agent logic construction.
- **[Deep-Learning-Toolkit](https://github.com/yuliu625/Yu-Deep-Learning-Toolkit)**: A foundation for general deep learning tasks.
- **[Data-Science-Toolkit](https://github.com/yuliu625/Yu-Data-Science-Toolkit)**: Tools for data science and preprocessing.

### Existing Agent Research
- [Simulate-the-Prisoners-Dilemma-with-Agents](https://github.com/yuliu625/Simulate-the-Prisoners-Dilemma-with-Agents): An early attempt based on AutoGen, researching the decision-making behavior of LLM Agents in simple game theory scenarios like the Prisoner's Dilemma.
- [World-of-Six](https://github.com/yuliu625/World-of-Six): Research on Agent decision-making behavior in environments with network effects (Paper accepted by SWAIB [2025]).

### Ongoing Projects (To be Open-Sourced)
- Research on the expected behavior of LLM-based Agents in environments with network effects (Code will be released after journal publication).
- A document intelligence project analyzing financial reports through a Multi-agent System.

