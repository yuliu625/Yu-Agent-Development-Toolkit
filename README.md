# Agent Development Toolkit

## Overview
This is a comprehensive, engineering-focused toolkit dedicated to **LLM-based Agent development**.

The core objective of this repository is to provide a set of battle-tested, highly reusable, and robust tools and components specifically for building and managing complex AI Agent systems. My engineering focus is placed on **Stability**, **Process Controllability**, and **System Scalability**, aiming to expedite the transition from rapid prototyping to production-grade Agent applications.

### Core Technology Stack
I've chosen and deeply customized the **LangChain** and **LangGraph** ecosystem as the foundation for building complex Agent systems:
- **LangChain**: Fully leveraging its **Runnable** and **Serializable** abstractions provides a solid base for constructing high-level, composable Agent components.
- **LangGraph**: Its core strength lies in the **state machine model**. This allows for precise control and flexible definition of the Agent's decision-making flow and loops, making it particularly well-suited for scenarios requiring complex planning, human-in-the-loop interaction, or tool use.

*Context Note: I previously experimented with Agent development based on Autogen and Llama-Index. However, practical experience has shown that the LangChain + LangGraph ecosystem offers greater advantages in terms of code engineering when dealing with complex Agent systems that demand fine-grained process control and high stability.*


## Repository Structure and Core Functional Modules
The toolkit is organized into several distinct functional modules to address common challenges in Agent development:

### General Core Utilities
Provides essential foundational tools that are framework-agnostic but indispensable in any LLM/Agent project, focusing on input/output reliability.
- **Structured Data Handling**: Includes tools for **safely extracting JSON**, processing structured input, and simplified structured data extraction. The primary focus is solving the issue of unstable LLM output format and difficult validation.
- **Content and Context Processing**: Offers VLM content block handling (for multimodal input construction) and a content tokenizer used for multi-agent identity or RAG metadata tagging.

### LangChain Extensions and Toolset
A series of components built upon the LangChain system to enhance LLM Agent stability and engineering efficiency.
- **`base_agent`**: A universal Agent superclass, pre-built with structured output, self-correction, and retry mechanisms.
- **Model Building and Management**: Unified **LLM Factory**, **Embedding Model Factory**, and a configuration builder, simplifying model switching and configuration management.
- **Prompt Engineering**: Encapsulates **`jinja2`-based Prompt loading and management**, and provides safe formatting methods to prevent risks like silent failures from template variable errors.
- **Utility Tools**: Includes tools for gracefully handling streaming, unified processing of Reasoning content, and safe message copying.

### LangGraph Flow Control and Graph Construction
Focused on utilizing LangGraph's state machine features to achieve precise control over the Agent workflow.
- **`graph_builder`**: Defines the computation graph construction standard, helping to formalize the definition of complex Agent loops and logic.
- **Utility Tools**: Aids in managing LangGraph-related states and nodes.

### LLM Launchers: Unified LLM Inference Service
A specialized module for uniformly managing and simplifying the use and deployment of various LLM inference services (both local and remote APIs). It aims to abstract away tedious service configuration details.

*Other Frameworks: You can view some of my earlier tools built on Autogen and Llama-Index in the `_other_agent_frameworks` directory.*


## More of My Agent and Deep Learning Projects
Feel free to check out my other repositories focused on specific domains or research to learn more about my engineering and research work. These projects, together with the Agent Development Toolkit, form my technological ecosystem.

### General Toolsets
- [Deep-Learning-Toolkit](https://github.com/yuliu625/Yu-Deep-Learning-Toolkit): A general-purpose toolset for deep learning tasks.
- [Data-Science-Toolkit](https://github.com/yuliu625/Yu-Data-Science-Toolkit): My toolkit built for data science tasks.

### Existing Agent Research Projects
- [Simulate-the-Prisoners-Dilemma-with-Agents](https://github.com/yuliu625/Simulate-the-Prisoners-Dilemma-with-Agents): My early attempt based on Autogen, investigating the decision-making behavior of LLM Agents in simple game-theory scenarios like the Prisoner's Dilemma.
- [World-of-Six](https://github.com/yuliu625/World-of-Six): My research on Agent decision-making behavior in environments with network effects. (Paper accepted by SWAIB [2025])

### Ongoing Projects (Future Open-Source)
- Research in progress on the anticipated behavior of LLM-based Agents in environments with network effects (code will be open-sourced after journal publication).
- Working on a document intelligence project that analyzes financial reports by building a Multi-Agent System.

