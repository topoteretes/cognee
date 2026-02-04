<div align="center">
  <a href="https://github.com/topoteretes/cognee">
    <img src="https://raw.githubusercontent.com/topoteretes/cognee/refs/heads/dev/assets/cognee-logo-transparent.png" alt="Cognee Logo" height="60">
  </a>

  <br />

  Cognee - Accurate and Persistent AI Memory

  <p align="center">
  <a href="https://www.youtube.com/watch?v=1bezuvLwJmw&t=2s">Demo</a>
  .
  <a href="https://docs.cognee.ai/">Docs</a>
  .
  <a href="https://cognee.ai">Learn More</a>
  ·
  <a href="https://discord.gg/NQPKmU5CCg">Join Discord</a>
  ·
  <a href="https://www.reddit.com/r/AIMemory/">Join r/AIMemory</a>
  .
  <a href="https://github.com/topoteretes/cognee-community">Community Plugins & Add-ons</a>
  </p>


  [![GitHub forks](https://img.shields.io/github/forks/topoteretes/cognee.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/topoteretes/cognee/network/)
  [![GitHub stars](https://img.shields.io/github/stars/topoteretes/cognee.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/topoteretes/cognee/stargazers/)
  [![GitHub commits](https://badgen.net/github/commits/topoteretes/cognee)](https://GitHub.com/topoteretes/cognee/commit/)
  [![GitHub tag](https://badgen.net/github/tag/topoteretes/cognee)](https://github.com/topoteretes/cognee/tags/)
  [![Downloads](https://static.pepy.tech/badge/cognee)](https://pepy.tech/project/cognee)
  [![License](https://img.shields.io/github/license/topoteretes/cognee?colorA=00C586&colorB=000000)](https://github.com/topoteretes/cognee/blob/main/LICENSE)
  [![Contributors](https://img.shields.io/github/contributors/topoteretes/cognee?colorA=00C586&colorB=000000)](https://github.com/topoteretes/cognee/graphs/contributors)
  <a href="https://github.com/sponsors/topoteretes"><img src="https://img.shields.io/badge/Sponsor-❤️-ff69b4.svg" alt="Sponsor"></a>

<p>
  <a href="https://www.producthunt.com/posts/cognee?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_souce=badge-cognee" target="_blank" style="display:inline-block; margin-right:10px;">
    <img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=946346&theme=light&period=daily&t=1744472480704" alt="cognee - Memory&#0032;for&#0032;AI&#0032;Agents&#0032;&#0032;in&#0032;5&#0032;lines&#0032;of&#0032;code | Product Hunt" width="250" height="54" />
  </a>

  <a href="https://trendshift.io/repositories/13955" target="_blank" style="display:inline-block;">
    <img src="https://trendshift.io/api/badge/repositories/13955" alt="topoteretes%2Fcognee | Trendshift" width="250" height="55" />
  </a>
</p>

Use your data to build personalized and dynamic memory for AI Agents. Cognee lets you replace RAG with scalable and modular ECL (Extract, Cognify, Load) pipelines.

  <p align="center">
  🌐 Available Languages
  :
  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=de">Deutsch</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=es">Español</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=fr">Français</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=ja">日本語</a> |
  <a href="README_ko.md">한국어</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=pt">Português</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=ru">Русский</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=zh">中文</a>
  </p>


<div style="text-align: center">
  <img src="https://raw.githubusercontent.com/topoteretes/cognee/refs/heads/main/assets/cognee_benefits.png" alt="Why cognee?" width="50%" />
</div>
</div>

## About Cognee

Cognee is an open-source tool and platform that transforms your raw data into persistent and dynamic AI memory for Agents. It combines vector search with graph databases to make your documents both searchable by meaning and connected by relationships.

You can use Cognee in two ways:

1. [Self-host Cognee Open Source](https://docs.cognee.ai/getting-started/installation), which stores all data locally by default.
2. [Connect to Cognee Cloud](https://platform.cognee.ai/), and get the same OSS stack on managed infrastructure for easier development and productionization.

### Cognee Open Source (self-hosted):

- Interconnects any type of data — including past conversations, files, images, and audio transcriptions
- Replaces traditional RAG systems with a unified memory layer built on graphs and vectors
- Reduces developer effort and infrastructure cost while improving quality and precision
- Provides Pythonic data pipelines for ingestion from 30+ data sources
- Offers high customizability through user-defined tasks, modular pipelines, and built-in search endpoints

### Cognee Cloud (managed):
- Hosted web UI dashboard
- Automatic version updates
- Resource usage analytics
- GDPR compliant, enterprise-grade security

## Basic Usage & Feature Guide

To learn more, [check out this short, end-to-end Colab walkthrough](https://colab.research.google.com/drive/12Vi9zID-M3fpKpKiaqDBvkk98ElkRPWy?usp=sharing) of Cognee's core features.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12Vi9zID-M3fpKpKiaqDBvkk98ElkRPWy?usp=sharing)

## Quickstart

Let’s try Cognee in just a few lines of code. For detailed setup and configuration, see the [Cognee Docs](https://docs.cognee.ai/getting-started/installation#environment-configuration).

### Prerequisites

- Python 3.10 to 3.13

### Step 1: Install Cognee

You can install Cognee with **pip**, **poetry**, **uv**, or your preferred Python package manager.

```bash
uv pip install cognee
```

### Step 2: Configure the LLM
```python
import os
os.environ["LLM_API_KEY"] = "YOUR OPENAI_API_KEY"
```
Alternatively, create a `.env` file using our [template](https://github.com/topoteretes/cognee/blob/main/.env.template).

To integrate other LLM providers, see our [LLM Provider Documentation](https://docs.cognee.ai/setup-configuration/llm-providers).

### Step 3: Run the Pipeline

Cognee will take your documents, generate a knowledge graph from them and then query the graph based on combined relationships.

Now, run a minimal pipeline:

```python
import cognee
import asyncio
from pprint import pprint


async def main():
    # Add text to cognee
    await cognee.add("Cognee turns documents into AI memory.")

    # Generate the knowledge graph
    await cognee.cognify()

    # Add memory algorithms to the graph
    await cognee.memify()

    # Query the knowledge graph
    results = await cognee.search("What does Cognee do?")

    # Display the results
    for result in results:
        pprint(result)


if __name__ == '__main__':
    asyncio.run(main())

```

As you can see, the output is generated from the document we previously stored in Cognee:

```bash
  Cognee turns documents into AI memory.
```

### Use the Cognee CLI

As an alternative, you can get started with these essential commands:

```bash
cognee-cli add "Cognee turns documents into AI memory."

cognee-cli cognify

cognee-cli search "What does Cognee do?"
cognee-cli delete --all

```

To open the local UI, run:
```bash
cognee-cli -ui
```

## Demos & Examples

See Cognee in action:

### Persistent Agent Memory

[Cognee Memory for LangGraph Agents](https://github.com/user-attachments/assets/e113b628-7212-4a2b-b288-0be39a93a1c3)

### Simple GraphRAG

[Watch Demo](https://github.com/user-attachments/assets/f2186b2e-305a-42b0-9c2d-9f4473f15df8)

### Cognee with Ollama

[Watch Demo](https://github.com/user-attachments/assets/39672858-f774-4136-b957-1e2de67b8981)


## Community & Support

### Contributing
We welcome contributions from the community! Your input helps make Cognee better for everyone. See [`CONTRIBUTING.md`](CONTRIBUTING.md) to get started.

### Code of Conduct

We're committed to fostering an inclusive and respectful community. Read our [Code of Conduct](https://github.com/topoteretes/cognee/blob/main/CODE_OF_CONDUCT.md) for guidelines.

## Research & Citation

We recently published a research paper on optimizing knowledge graphs for LLM reasoning:

```bibtex
@misc{markovic2025optimizinginterfaceknowledgegraphs,
      title={Optimizing the Interface Between Knowledge Graphs and LLMs for Complex Reasoning},
      author={Vasilije Markovic and Lazar Obradovic and Laszlo Hajdu and Jovan Pavlovic},
      year={2025},
      eprint={2505.24478},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.24478},
}
```
