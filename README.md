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

[//]: # (<p>)

[//]: # (  <b>Share cognee Repository</b>)

[//]: # (</p>)

[//]: # (<p>)

[//]: # (  <a href="https://twitter.com/intent/tweet?text=Check%20this%20GitHub%20repository%20out.%20Cognee%20-%20Let%27s%20you%20easily%20build,%20manage%20and%20run%20useful%20autonomous%20AI%20agents.&url=https://github.com/topoteretes/cognee&hashtags=AGI,Autonomics,Cognee,future" target="_blank">)

[//]: # (    <img src="https://img.shields.io/twitter/follow/_promethAI?label=Share Repo on Twitter&style=social" alt="Follow Cognee"/>)

[//]: # (  </a>)

[//]: # (  <a href="https://t.me/share/url?text=Check%20this%20GitHub%20repository%20out.%20Cognee%20-%20Let%27s%20you%20easily%20build,%20manage%20and%20run%20useful%20autonomous%20AI%20agents.&url=https://github.com/topoteretes/cognee" target="_blank">)

[//]: # (    <img src="https://img.shields.io/twitter/url?label=Telegram&logo=Telegram&style=social&url=https://github.com/topoteretes/cognee" alt="Share on Telegram"/>)

[//]: # (  </a>)

[//]: # (  <a href="https://api.whatsapp.com/send?text=Check%20this%20GitHub%20repository%20out.%20Cognee%20-%20Let's%20you%20easily%20build,%20manage%20and%20run%20useful%20autonomous%20AI%20agents.%20https://github.com/topoteretes/cognee" target="_blank">)

[//]: # (    <img src="https://img.shields.io/twitter/url?label=whatsapp&logo=whatsapp&style=social&url=https://github.com/topoteretes/cognee" />)

[//]: # (  </a>)

[//]: # (  <a href="https://www.reddit.com/submit?url=https://github.com/topoteretes/cognee&title=Check%20this%20GitHub%20repository%20out.%20Cognee%20-%20Let's%20you%20easily%20build,%20manage%20and%20run%20useful%20autonomous%20AI%20agents.)

[//]: # (  " target="_blank">)

[//]: # (    <img src="https://img.shields.io/twitter/url?label=Reddit&logo=Reddit&style=social&url=https://github.com/topoteretes/cognee" alt="Share on Reddit"/>)

[//]: # (  </a>)

[//]: # (  <a href="mailto:?subject=Check%20this%20GitHub%20repository%20out.&body=Cognee%20-%20Let%27s%20you%20easily%20build,%20manage%20and%20run%20useful%20autonomous%20AI%20agents.%3A%0Ahttps://github.com/topoteretes/cognee" target="_blank">)

[//]: # (    <img src="https://img.shields.io/twitter/url?label=Gmail&logo=Gmail&style=social&url=https://github.com/topoteretes/cognee"/>)

[//]: # (  </a>)

[//]: # (  <a href="https://www.buymeacoffee.com/promethAI" target="_blank">)

[//]: # (    <img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="23" width="100" style="border-radius:1px">)

[//]: # (  </a>)

[//]: # (</p>)

[//]: # ()
[//]: # (<hr>)

[//]: # ()
[//]: # ([Star us on Github!]&#40;https://www.github.com/topoteretes/cognee&#41;)

[//]: # ()
[//]: # (<a href="https://www.cognee.ai" target="_blank">Cognee</a> runs in iterations, from POC towards production-ready code.)


## 🚀 It's alive
<p>

Try it yourself on Whatsapp with one of our <a href="https://keepi.ai">partners</a> by typing `/save {content you want to save}` followed by `/query {knowledge you saved previously}`
</p>


## 📦 Installation

With pip:

```bash
pip install cognee
```

With poetry:

```bash
poetry add cognee
```

## 💻 Usage

Check out our demo notebook [here](cognee%20-%20Get%20Started.ipynb)


- Add a new piece of information to storage
```
import cognee
cognee.add(absolute_data_path, dataset_name)
```

- Use LLMs and cognee to create graphs
 
``` 
cognee.cognify(dataset_name)
 ``` 

- Render the graph after adding your Graphistry credentials to .env

```
graph_url = await render_graph(graph, graph_type = "networkx")
print(graph_url)
```

- Query the graph for a piece of information

```
query_params = {
    SearchType.SIMILARITY: {'query': 'your search query here'}
}
cognee.search(graph, query_params) 
``` 


## Demo

[<img src="https://i3.ytimg.com/vi/yjParvJVgPI/maxresdefault.jpg" width="100%">](https://www.youtube.com/watch?v=yjParvJVgPI "Learn about cognee: 55")

## Architecture

### How Cognee Enhances Your Contextual Memory

Our framework for the OpenAI, Graph (Neo4j) and Vector (Weaviate) databases introduces three key enhancements:

- Query Classifiers: Navigate information graph using Pydantic OpenAI classifiers.
- Document Topology: Structure and store documents in public and private domains.
- Personalized Context: Provide a context object to the LLM for a better response.


![Image](assets/architecture.png)

