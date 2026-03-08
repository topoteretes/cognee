<div align="center">
  <a href="https://github.com/topoteretes/cognee">
    <img src="https://raw.githubusercontent.com/topoteretes/cognee/refs/heads/dev/assets/cognee-logo-transparent.png" alt="Cognee Logo" height="60">
  </a>

  <br />

  Cognee - Build AI memory with a Knowledge Engine that learns

  <p align="center">
  <a href="https://www.youtube.com/watch?v=8hmqS2Y5RVQ&t=13s">Demo</a>
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
  <a href="https://trendshift.io/repositories/13955" target="_blank" style="display:inline-block;">
    <img src="https://trendshift.io/api/badge/repositories/13955" alt="topoteretes%2Fcognee | Trendshift" width="250" height="55" />
  </a>
</p>

Use our knowledge engine to build personalized and dynamic memory for AI Agents.

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
  <img src="https://raw.githubusercontent.com/topoteretes/cognee/refs/heads/main/assets/cognee_benefits.png" alt="Why cognee?" width="80%" />
</div>
</div>




## About Cognee

Cognee is an open-source knowledge engine that lets you ingest data in any format or structure and continuously learns to provide the right context for AI agents. It combines vector search, graph databases and cognitive science approaches to make your documents both searchable by meaning and connected by relationships as they change and evolve.



:star: _Help us reach more developers and grow the cognee community. Star this repo!_


### Why use Cognee:

- Knowledge infrastructure — unified ingestion, graph/vector search, runs locally, ontology grounding, multimodal
- Persistent and Learning Agents - learn from feedback, context management, cross-agent knowledge sharing
- Reliable and Trustworthy Agents - agentic user/tenant isolation, traceability, OTEL collector, audit traits



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

Cognee will take your documents, load them into the knowledge angine and search combined vector/graph relationships.

Now, run a minimal pipeline:

```python
import cognee
import asyncio
from pprint import pprint


async def main():
    # Add text to cognee
    await cognee.add("Cognee turns documents into AI memory.")

    # Add to knowledge engine
    await cognee.cognify()

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

[![Watch Demo](https://img.youtube.com/vi/8hmqS2Y5RVQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=8hmqS2Y5RVQ&t=13s)


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
