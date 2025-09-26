<div align="center">
  <a href="https://github.com/topoteretes/cognee">
    <img src="https://raw.githubusercontent.com/topoteretes/cognee/refs/heads/dev/assets/cognee-logo-transparent.png" alt="Cognee Logo" height="60">
  </a>

  <br />

  cognee - Memory for AI Agents in 6 lines of code

  <p align="center">
  <a href="https://www.youtube.com/watch?v=1bezuvLwJmw&t=2s">Demo</a>
  .
  <a href="https://cognee.ai">Learn more</a>
  ·
  <a href="https://discord.gg/NQPKmU5CCg">Join Discord</a>
  ·
  <a href="https://www.reddit.com/r/AIMemory/">Join r/AIMemory</a>
  .
  <a href="https://docs.cognee.ai/">Docs</a>
  .
  <a href="https://github.com/topoteretes/cognee-community">cognee community repo</a>
  </p>


  [![GitHub forks](https://img.shields.io/github/forks/topoteretes/cognee.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/topoteretes/cognee/network/)
  [![GitHub stars](https://img.shields.io/github/stars/topoteretes/cognee.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/topoteretes/cognee/stargazers/)
  [![GitHub commits](https://badgen.net/github/commits/topoteretes/cognee)](https://GitHub.com/topoteretes/cognee/commit/)
  [![Github tag](https://badgen.net/github/tag/topoteretes/cognee)](https://github.com/topoteretes/cognee/tags/)
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





Build dynamic memory for Agents and replace RAG using scalable, modular ECL (Extract, Cognify, Load) pipelines.

  <p align="center">
  🌐 Available Languages
  :
  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=de">Deutsch</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=es">Español</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=fr">français</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=ja">日本語</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=ko">한국어</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=pt">Português</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=ru">Русский</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=zh">中文</a>
  </p>


<div style="text-align: center">
  <img src="https://raw.githubusercontent.com/topoteretes/cognee/refs/heads/main/assets/cognee_benefits.png" alt="Why cognee?" width="50%" />
</div>
</div>



## Get Started

Get started quickly with a Google Colab  <a href="https://colab.research.google.com/drive/1jHbWVypDgCLwjE71GSXhRL3YxYhCZzG1?usp=sharing">notebook</a> , <a href="https://deepnote.com/workspace/cognee-382213d0-0444-4c89-8265-13770e333c02/project/cognee-demo-78ffacb9-5832-4611-bb1a-560386068b30/notebook/Notebook-1-75b24cda566d4c24ab348f7150792601?utm_source=share-modal&utm_medium=product-shared-content&utm_campaign=notebook&utm_content=78ffacb9-5832-4611-bb1a-560386068b30">Deepnote notebook</a> or  <a href="https://github.com/topoteretes/cognee/tree/main/cognee-starter-kit">starter repo</a>


## About cognee

Self-hosted package:

- Interconnects any kind of documents: past conversations, files, images, and audio transcriptions
- Replaces RAG systems with a memory layer based on graphs and vectors
- Reduces developer effort and cost, while increasing quality and precision
- Provides Pythonic data pipelines that manage data ingestion from 30+ data sources
- Is highly customizable with custom tasks, pipelines, and a set of built-in search endpoints

Hosted platform:
- Includes a managed UI and a [hosted solution](www.cognee.ai)



## Self-Hosted (Open Source)


### 📦 Installation

You can install Cognee using either **pip**, **poetry**, **uv** or any other python package manager.

Cognee supports Python 3.10 to 3.12

#### With uv

```bash
uv pip install cognee
```

Detailed instructions can be found in our [docs](https://docs.cognee.ai/getting-started/installation#environment-configuration)

### 💻 Basic Usage

#### Setup

```
import os
os.environ["LLM_API_KEY"] = "YOUR OPENAI_API_KEY"

```

You can also set the variables by creating .env file, using our <a href="https://github.com/topoteretes/cognee/blob/main/.env.template">template.</a>
To use different LLM providers, for more info check out our <a href="https://docs.cognee.ai/setup-configuration/llm-providers">documentation</a>


#### Simple example



##### Python

This script will run the default pipeline:

```python
import cognee
import asyncio


async def main():
    # Add text to cognee
    await cognee.add("Cognee turns documents into AI memory.")

    # Generate the knowledge graph
    await cognee.cognify()

    # Add memory algorithms to the graph
    await cognee.memify()

    # Query the knowledge graph
    results = await cognee.search("What does cognee do?")

    # Display the results
    for result in results:
        print(result)


if __name__ == '__main__':
    asyncio.run(main())

```
Example output:
```
  Cognee turns documents into AI memory.

```
##### Via CLI

Let's get the basics covered

```
cognee-cli add "Cognee turns documents into AI memory."

cognee-cli cognify

cognee-cli search "What does cognee do?"
cognee-cli delete --all

```
or run
```
cognee-cli -ui
```


</div>


### Hosted Platform

Get up and running in minutes with automatic updates, analytics, and enterprise security.

1. Sign up on [cogwit](https://www.cognee.ai)
2. Add your API key to local UI and sync your data to Cogwit




## Demos

1. Cogwit Beta demo:

[Cogwit Beta](https://github.com/user-attachments/assets/fa520cd2-2913-4246-a444-902ea5242cb0)

2. Simple GraphRAG demo

[Simple GraphRAG demo](https://github.com/user-attachments/assets/d80b0776-4eb9-4b8e-aa22-3691e2d44b8f)

3. cognee with Ollama

[cognee with local models](https://github.com/user-attachments/assets/8621d3e8-ecb8-4860-afb2-5594f2ee17db)


## Contributing
Your contributions are at the core of making this a true open source project. Any contributions you make are **greatly appreciated**. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for more information.


## Code of Conduct

We are committed to making open source an enjoyable and respectful experience for our community. See <a href="https://github.com/topoteretes/cognee/blob/main/CODE_OF_CONDUCT.md"><code>CODE_OF_CONDUCT</code></a> for more information.

## Citation

We now have a paper you can cite:

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
