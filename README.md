# cognee

Deterministic LLMs Outputs for AI Engineers using graphs, LLMs and vector retrieval


<p>
  <a href="https://cognee.ai" target="_blank">
    <img src="https://raw.githubusercontent.com/topoteretes/cognee/main/assets/cognee-logo.png" width="160px" alt="Cognee logo" />
  </a>
</p>


<p>
  <i>Open-source framework for creating self-improving deterministic outputs for LLMs.</i>
</p>

<p>
  <a href="https://github.com/topoteretes/cognee/fork">
    <img src="https://img.shields.io/github/forks/topoteretes/cognee?style=for-the-badge" alt="cognee forks"/>
  </a>
  <a href="https://github.com/topoteretes/cognee/stargazers">
    <img src="https://img.shields.io/github/stars/topoteretes/cognee?style=for-the-badge" alt="cognee stars"/>
  </a>
  <a href="https://github.com/topoteretes/cognee/pulls">
    <img src="https://img.shields.io/github/issues-pr/topoteretes/cognee?style=for-the-badge" alt="cognee pull-requests"/>
  </a>
  <a href="https://github.com/topoteretes/cognee/releases">
    <img src="https://img.shields.io/github/release/topoteretes/cognee?&label=Latest&style=for-the-badge" alt="cognee releases" />
  </a>
</p>

![Cognee Demo](assets/cognee_demo.gif)


Try it in a Google collab  <a href="https://colab.research.google.com/drive/184Kpe9XGjrt8nVss0WDiPsPRrPk5lZ6o?usp=sharing">notebook</a>  or have a look at our <a href="https://topoteretes.github.io/cognee">documentation</a>

If you have questions, join our  <a href="https://discord.gg/NQPKmU5CCg">Discord</a> community





## 📦 Installation

### With pip

```bash
pip install cognee
```


### With poetry

```bash
poetry add cognee
```


## 💻 Usage

### Setup

```
import os

os.environ["LLM_API_KEY"] = "YOUR OPENAI_API_KEY"

```
or 
```
import cognee
cognee.config.llm_api_key = "YOUR_OPENAI_API_KEY"
```

You can also use Ollama or Anyscale as your LLM provider. For more info on local models check our [docs](https://topoteretes.github.io/cognee)

### Run

```
import cognee

text = """Natural language processing (NLP) is an interdisciplinary
       subfield of computer science and information retrieval"""

cognee.add([text], "example_dataset") # Add a new piece of information

cognee.cognify() # Use LLMs and cognee to create knowledge

search_results = cognee.search("SIMILARITY", "computer science") # Query cognee for the knowledge

for result_text in search_results[0]:
    print(result_text)

```
Add alternative data types:
```
cognee.add("file://{absolute_path_to_file}", dataset_name)
```
Or
```
cognee.add("data://{absolute_path_to_directory}", dataset_name)

# This is useful if you have a directory with files organized in subdirectories.
# You can target which directory to add by providing dataset_name.
# Example:
#            root
#           /    \
#      reports  bills
#     /       \
#   2024     2023
#
# cognee.add("data://{absolute_path_to_root}", "reports.2024")
# This will add just directory 2024 under reports.
```

Read more [here](docs/index.md#run).

## Vector retrieval, Graphs and LLMs

Cognee supports a variety of tools and services for different operations:

- **Local Setup**: By default, LanceDB runs locally with NetworkX and OpenAI.

- **Vector Stores**: Cognee supports Qdrant and Weaviate for vector storage.

- **Language Models (LLMs)**: You can use either Anyscale or Ollama as your LLM provider.

- **Graph Stores**: In addition to LanceDB, Neo4j is also supported for graph storage.

## Demo

Check out our demo notebook [here](https://github.com/topoteretes/cognee/blob/main/notebooks/cognee%20-%20Get%20Started.ipynb)



[<img src="https://i3.ytimg.com/vi/-ARUfIzhzC4/maxresdefault.jpg" width="100%">](https://www.youtube.com/watch?v=BDFt4xVPmro "Learn about cognee: 55")





## How it works





![Image](assets/architecture.png)


## Star History


[![Star History Chart](https://api.star-history.com/svg?repos=topoteretes/cognee&type=Date)](https://star-history.com/#topoteretes/cognee&Date)
