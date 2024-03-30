# cognee

Make data processing for LLMs easy


<p>
  <a href="https://cognee.ai" target="_blank">
    <img src="assets/cognee-logo.png" width="160px" alt="Cognee logo" />
  </a>
</p>

<p>
  <i>Open-source framework for creating knowledge graphs and data models for LLMs.</i>
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


## 🚀 It's alive

<p>
Try it yourself on Whatsapp with one of our <a href="https://keepi.ai" target="_blank">partners</a> by typing `/save {content you want to save}` followed by `/query {knowledge you saved previously}`
For more info here are the <a href="https://topoteretes.github.io/cognee">docs</a>
</p>


## 📦 Installation

With pip:

```bash
pip install "cognee[weaviate]"
```

With poetry:

```bash
poetry add "cognee[weaviate]"
```

## 💻 Usage

### Setup

Create `.env` file in your project in order to store environment variables such as API keys.

Note: Don't push `.env` file to git repo as it will expose those keys to others.

If cognee is installed with Weaviate as a vector database provider, add Weaviate environment variables.
```
WEAVIATE_URL = {YOUR_WEAVIATE_URL}
WEAVIATE_API_KEY = {YOUR_WEAVIATE_API_KEY}
```

Otherwise if cognee is installed with a default (Qdrant) vector database provider, add Qdrant environment variables.
```
QDRANT_URL = {YOUR_QDRANT_URL}
QDRANT_API_KEY = {YOUR_QDRANT_API_KEY}
```

Add OpenAI API Key environment variable
```
OPENAI_API_KEY = {YOUR_OPENAI_API_KEY}
```

Cognee stores data and system files inside the library directory, which is lost if the library folder is removed.
You can change the directories where cognee will store data and system files by calling config functions.
```
import cognee

cognee.config.system_root_directory(absolute_path_to_directory)

cognee.config.data_root_directory(absolute_path_to_directory)
```

### Run

Add a new piece of information to storage
```
import cognee

cognee.add("some_text", dataset_name)

cognee.add([
    "some_text_1",
    "some_text_2",
    "some_text_3",
    ...
])
```
Or
```
cognee.add("file://{absolute_path_to_file}", dataset_name)

cognee.add(
    [
        "file://{absolute_path_to_file_1}",
        "file://{absolute_path_to_file_2}",
        "file://{absolute_path_to_file_3}",
        ...
    ],
    dataset_name
)
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

Use LLMs and cognee to create graphs
``` 
cognee.cognify(dataset_name)
 ``` 

Render the graph with our util function

```
from cognee.utils import render_graph

graph_url = await render_graph(graph)

print(graph_url)
```

Query the graph for a piece of information
```
query_params = {
    SearchType.SIMILARITY: {'query': 'your search query here'}
}

search_results = cognee.search(graph, query_params)

print(search_results)
```


## Demo

Check out our demo notebook [here](https://github.com/topoteretes/cognee/blob/main/notebooks/cognee%20-%20Get%20Started.ipynb)


## Architecture

[<img src="https://i3.ytimg.com/vi/-ARUfIzhzC4/maxresdefault.jpg" width="100%">](https://youtu.be/-ARUfIzhzC4 "Learn about cognee: 55")


### How Cognee Enhances Your Contextual Memory

Our framework for the OpenAI, Graph (Neo4j) and Vector (Weaviate) databases introduces three key enhancements:

- Query Classifiers: Navigate information graph using Pydantic OpenAI classifiers.
- Document Topology: Structure and store documents in public and private domains.
- Personalized Context: Provide a context object to the LLM for a better response.


![Image](assets/architecture.png)

