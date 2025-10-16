<div align="center">
  <a href="https://github.com/topoteretes/cognee">
    <img src="https://raw.githubusercontent.com/topoteretes/cognee/refs/heads/dev/assets/cognee-logo-transparent.png" alt="Cognee Logo" height="60">
  </a>
  <h1>cognee - Memory for AI Agents in 6 lines of code</h1>
  <p align="center">
  <a href="https://www.youtube.com/watch?v=1bezuvLwJmw&t=2s">Demo</a>
  ·
  <a href="https://cognee.ai">Learn more</a>
  ·
  <a href="https://discord.gg/NQPKmU5CCg">Join Discord</a>
  ·
  <a href="https://www.reddit.com/r/AIMemory/">Join r/AIMemory</a>
  ·
  <a href="https://docs.cognee.ai/">Docs</a>
  ·
  <a href="https://github.com/topoteretes/cognee-community">cognee community repo</a>
  </p>
  
  [![GitHub forks](https://img.shields.io/github/forks/topoteretes/cognee.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/topoteretes/cognee/network/)

</div>


<div align="center">
  <a href="https://cognee.ai" target="_blank">
    <img src="https://raw.githubusercontent.com/topoteretes/cognee/refs/heads/dev/assets/gifs/cognee_demo_small.gif">
  </a>
</div>


## What is cognee?
**cognee** is a memory layer for AI applications. It offers production-ready deterministic context retrieval based on data graphs
and LLM agents.

**TL;DR** Cognee is like an OS for AI systems - it gives context and memory to your AI.

## Quickstart - Get Started in 3 lines of code

- add your data with `cognee.add()`
- cognify your data with `cognee.cognify()` to turn unstructured text into structured knowledge graphs
- search with `cognee.search()` to retrieve contextually relevant information


Example:

```python
import cognee
import os

os.environ["LLM_API_KEY"] = "YOUR OPENAI_API_KEY"

cognee.config.llm_api_key = os.environ.get("LLM_API_KEY", "")

await cognee.add("https://www.youtube.com/watch?v=uge8XbjEde0") # Add data

await cognee.cognify() # Get knowledge graph

# Query cognee
query_results = await cognee.search("UNIQUE", "What is the nature of light?")

for result in query_results:
    print(result)

```


## About cognee

- deterministic **graph-based RAG** - Get insights from connected, structured data
- **Build LLM systems backed by graphs** - Create intelligent systems that maintain context and relationships across vast amounts of information
- **make better systems with your own graphs** - Use graph architecture as the foundation of your AI systems

One of the biggest issues with most current RAG systems is that they rely purely on semantic search, which tends to miss important context because it's just matching words. By using a **graph-based system**, we can keep the relationships between pieces of information intact, so your AI agents get the full picture and make smarter, more connected decisions.

Instead of just doing semantic search on chunks, cognee lets you:
- **Create layered graph architectures** from unstructured text that explicitly map relationships between entities, concepts, and topics
- **Build custom knowledge bases** that work alongside or instead of vector search
- **Query your data as a graph** using semantic, keyword, or cypher queries to get precise, context-aware answers
- **Give your AI agents memory and context** so they understand not just keywords but the actual connections between data points


## Installation

### 1. Install cognee:

```bash
pip install cognee
```

### 2. Set up environment variables

Create a .env file and configure it with your LLM provider API keys:

```bash
LLM_API_KEY="<YOUR_OPENAI_API_KEY>"
```

For alternative LLM providers (Ollama, Anthropic, etc), refer to [our LLM configuration guide](https://docs.cognee.ai/llms/overview/).


## Basic Usage

### Working With Text

```python
import cognee
import os
os.environ["LLM_API_KEY"] = "YOUR OPENAI_API_KEY"

cognee.config.llm_api_key = os.environ.get("LLM_API_KEY", "")

# Add text data
text = """Natural language processing (NLP) is an interdisciplinary
        subfield of computer science and information retrieval."""
        
await cognee.add([text], "example_dataset") # Add data to cognee
await cognee.cognify() # Create knowledge graph

# Query
search_results = await cognee.search("INSIGHTS", query_text="Tell me about NLP")
```

### Working With Data Files

```python
import cognee
import os

os.environ["LLM_API_KEY"] = "YOUR OPENAI_API_KEY"
cognee.config.llm_api_key = os.environ.get("LLM_API_KEY", "")

# Add a file
await cognee.add("path/to/your/file.txt", "example_dataset")
await cognee.cognify()

# Search
search_results = await cognee.search("INSIGHTS", query_text="Summarize the key points")
```

## Codegraph Pipeline - Language Support

### Supported Languages

The cognee codegraph pipeline now supports multiple programming languages for code analysis:

- **Python** - Full AST parsing and code structure analysis
- **C#** - Class, method, and namespace extraction with tree-sitter
- **C++** - Function, class, and template analysis with tree-sitter

### C# Code Analysis Example

```python
import cognee
import os

os.environ["LLM_API_KEY"] = "YOUR_OPENAI_API_KEY"
cognee.config.llm_api_key = os.environ.get("LLM_API_KEY", "")

# Add C# code file
await cognee.add("path/to/your/code.cs", "csharp_project")
await cognee.cognify()

# Query C# code structure
search_results = await cognee.search(
    "INSIGHTS", 
    query_text="What classes and methods are defined in this codebase?"
)

for result in search_results:
    print(result)
```

### C++ Code Analysis Example

```python
import cognee
import os

os.environ["LLM_API_KEY"] = "YOUR_OPENAI_API_KEY"
cognee.config.llm_api_key = os.environ.get("LLM_API_KEY", "")

# Add C++ code file
await cognee.add("path/to/your/code.cpp", "cpp_project")
await cognee.cognify()

# Query C++ code structure
search_results = await cognee.search(
    "INSIGHTS", 
    query_text="What are the main functions and classes in this C++ code?"
)

for result in search_results:
    print(result)
```

### Test Coverage

The C# and C++ support includes comprehensive test coverage:

- **C# Tests**:
  - Class declaration parsing
  - Method extraction (including constructors, properties, async methods)
  - Namespace handling
  - Basic inheritance detection
  - Verified with sample code including Calculator class with multiple method types

- **C++ Tests**:
  - Function declaration parsing
  - Class structure extraction
  - Template detection
  - Namespace handling
  - Verified with sample code including vector operations and template classes

All tests validate that the tree-sitter parsers correctly extract code structures and that the codegraph pipeline processes C# (.cs) and C++ (.cpp) files alongside existing Python support.

## Hosted Platform

Try our **[hosted platform](https://platform.cognee.ai/)** which features:
- No-code interface for building knowledge graphs
- Pre-built integrations with popular data sources
- Team collaboration features
- Scalable infrastructure

## Demos
- [**cognee-demo**](https://github.com/topoteretes/cognee-demo) - Build a simple AI Assistant with memory using Cognee and CrewAI
- [**Cognee with LlamaIndex**](https://github.com/topoteretes/cognee-llamaindex-demo) - Add memory to LlamaIndex using cognee


## Documentation

For more detailed information, check out our [documentation](https://docs.cognee.ai/).

## Contributing

We love contributions! Check out our [Contributing Guide](CONTRIBUTING.md) for more information.

## Join Our Community

- [Discord](https://discord.gg/NQPKmU5CCg) - Chat with the community
- [Reddit r/AIMemory](https://www.reddit.com/r/AIMemory/) - Discuss AI memory systems
- [GitHub Discussions](https://github.com/topoteretes/cognee/discussions) - Ask questions and share ideas

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use cognee in your research, please cite it:

```bibtex
@software{cognee2024,
  author = {Cognee Team},
  title = {cognee: Memory Layer for AI Applications},
  year = {2024},
  url = {https://github.com/topoteretes/cognee}
}
```
