<div align="center">
  <a href="https://github.com/topoteretes/cognee">
    <img src="https://raw.githubusercontent.com/topoteretes/cognee/refs/heads/dev/assets/cognee-logo-transparent.png" alt="Cognee Logo" height="60">
  </a>

  <br />

  Cognee - Graph and Vector Memory for AI Agents

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
  <a href="https://github.com/topoteretes/cognee-community">Integrations</a>
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


Persistent and accurate memory for AI agents. With Cognee, your AI agent understands, reasons, and adapts.

  <p align="center">
  🌐 Available Languages
  :
  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=de">Deutsch</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=es">Español</a> |
  <a href="https://www.readme-i18n.com/topoteretes/cognee?lang=fr">Français</a> |
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



## Quickstart

- 🚀 Try it now on [Google Colab](https://colab.research.google.com/drive/12Vi9zID-M3fpKpKiaqDBvkk98ElkRPWy?usp=sharing)
- 📓 Explore our [Deepnote Notebook](https://deepnote.com/workspace/cognee-382213d0-0444-4c89-8265-13770e333c02/project/cognee-demo-78ffacb9-5832-4611-bb1a-560386068b30/notebook/Notebook-1-75b24cda566d4c24ab348f7150792601?utm_source=share-modal&utm_medium=product-shared-content&utm_campaign=notebook&utm_content=78ffacb9-5832-4611-bb1a-560386068b30)
- 🛠️ Clone our [Starter Repo](https://github.com/topoteretes/cognee/tree/main/cognee-starter-kit)


## About Cognee

Cognee transforms your data into a living knowledge graph that learns from feedback and auto-tunes to deliver better answers over time.

**Run anywhere:**
- 🏠 **Self-Hosted**: Runs locally, data stays on your device
- ☁️ **Cognee Cloud**: Same open-source Cognee, deployed on Modal for seamless workflows

**Self-Hosted Package:**

- Unified memory for all your data sources
- Domain-smart copilots that learn and adapt over time
- Flexible memory architecture for AI agents and devices
- Integrates easily with your current technology stack
- Pythonic data pipelines supporting 30+ data sources out of the box
- Fully extensible: customize tasks, pipelines, and search endpoints

**Cognee Cloud:**
- Get a managed UI and [Hosted Infrastructure](https://www.cognee.ai) with zero setup

## Self-Hosted (Open Source)

Run Cognee on your stack. Cognee integrates easily with your current technologies. See our [integration guides](https://docs.cognee.ai/setup-configuration/overview).


### 📦 Installation

Install Cognee with **pip**, **poetry**, **uv**, or your preferred Python package manager.

**Requirements:** Python 3.10 to 3.12

#### Using uv

```bash
uv pip install cognee
```

For detailed setup instructions, see our [Documentation](https://docs.cognee.ai/getting-started/installation#environment-configuration).

### 💻 Usage

#### Configuration

```python
import os
os.environ["LLM_API_KEY"] = "YOUR OPENAI_API_KEY"
```

Alternatively, create a `.env` file using our [template](https://github.com/topoteretes/cognee/blob/main/.env.template).
To integrate other LLM providers, see our [LLM Provider Documentation](https://docs.cognee.ai/setup-configuration/llm-providers).


#### Python Example

Run the default pipeline with this script:

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

#### CLI Example

Get started with these essential commands:

```
cognee-cli add "Cognee turns documents into AI memory."

cognee-cli cognify

cognee-cli search "What does cognee do?"
cognee-cli delete --all

```
Or run:
```
cognee-cli -ui
```

## Cognee Cloud

Cognee is the fastest way to start building reliable AI agent memory. Deploy in minutes with automatic updates, analytics, and enterprise-grade security.

- Sign up on [Cognee Cloud](https://www.cognee.ai)
- Add your API key to local UI and sync your data to Cognee Cloud
- Start building with managed infrastructure and zero configuration

## Trusted in Production

From regulated industries to startup stacks, Cognee is deployed in production and delivering value now. Read our [case studies](https://cognee.ai/blog) to learn more.

## Demos & Examples

See Cognee in action:

### Cognee Cloud Beta Demo

[Watch Demo](https://github.com/user-attachments/assets/fa520cd2-2913-4246-a444-902ea5242cb0)

### Simple GraphRAG Demo

[Watch Demo](https://github.com/user-attachments/assets/d80b0776-4eb9-4b8e-aa22-3691e2d44b8f)

### Cognee with Ollama

[Watch Demo](https://github.com/user-attachments/assets/8621d3e8-ecb8-4860-afb2-5594f2ee17db)


## Community & Support

### Contributing
We welcome contributions from the community! Your input helps make Cognee better for everyone. See [`CONTRIBUTING.md`](CONTRIBUTING.md) to get started.

### Code of Conduct

We're committed to fostering an inclusive and respectful community. Read our [Code of Conduct](https://github.com/topoteretes/cognee/blob/main/CODE_OF_CONDUCT.md) for guidelines.

## Research & Citation

Cite our research paper on optimizing knowledge graphs for LLM reasoning:

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
