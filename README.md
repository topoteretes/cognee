# cognee

AI Applications and RAGs - Cognitive Architecture, Testability, Production Ready Apps

<p>
  <a href="https://cognee.ai" target="_blank">
    <img src="assets/cognee-logo.png" width="160px" alt="Cognee logo" />
  </a>
</p>

<p>
  <i>Open-source framework for building and testing RAGs and Cognitive Architectures, designed for accuracy, transparency, and control.</i>
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
    <img src="https://img.shields.io/github/release/topoteretes/cognee?&label=Latest&style=for-the-badge" alt="cognee releases>
  </a>
</p>

<p>
  <b>Share cognee Repository</b>
</p>
<p>
  <a href="https://twitter.com/intent/tweet?text=Check%20this%20GitHub%20repository%20out.%20Cognee%20-%20Let%27s%20you%20easily%20build,%20manage%20and%20run%20useful%20autonomous%20AI%20agents.&url=https://github.com/topoteretes/cognee&hashtags=AGI,Autonomics,Cognee,future" target="_blank">
    <img src="https://img.shields.io/twitter/follow/_promethAI?label=Share Repo on Twitter&style=social" alt="Follow Cognee"/>
  </a>
  <a href="https://t.me/share/url?text=Check%20this%20GitHub%20repository%20out.%20Cognee%20-%20Let%27s%20you%20easily%20build,%20manage%20and%20run%20useful%20autonomous%20AI%20agents.&url=https://github.com/topoteretes/cognee" target="_blank">
    <img src="https://img.shields.io/twitter/url?label=Telegram&logo=Telegram&style=social&url=https://github.com/topoteretes/cognee" alt="Share on Telegram"/>
  </a>
  <a href="https://api.whatsapp.com/send?text=Check%20this%20GitHub%20repository%20out.%20Cognee%20-%20Let's%20you%20easily%20build,%20manage%20and%20run%20useful%20autonomous%20AI%20agents.%20https://github.com/topoteretes/cognee" target="_blank">
    <img src="https://img.shields.io/twitter/url?label=whatsapp&logo=whatsapp&style=social&url=https://github.com/topoteretes/cognee" />
  </a>
  <a href="https://www.reddit.com/submit?url=https://github.com/topoteretes/cognee&title=Check%20this%20GitHub%20repository%20out.%20Cognee%20-%20Let's%20you%20easily%20build,%20manage%20and%20run%20useful%20autonomous%20AI%20agents.
  " target="_blank">
    <img src="https://img.shields.io/twitter/url?label=Reddit&logo=Reddit&style=social&url=https://github.com/topoteretes/cognee" alt="Share on Reddit"/>
  </a>
  <a href="mailto:?subject=Check%20this%20GitHub%20repository%20out.&body=Cognee%20-%20Let%27s%20you%20easily%20build,%20manage%20and%20run%20useful%20autonomous%20AI%20agents.%3A%0Ahttps://github.com/topoteretes/cognee" target="_blank">
    <img src="https://img.shields.io/twitter/url?label=Gmail&logo=Gmail&style=social&url=https://github.com/topoteretes/cognee"/>
  </a>
  <a href="https://www.buymeacoffee.com/promethAI" target="_blank">
    <img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="23" width="100" style="border-radius:1px">
  </a>
</p>

<hr>

[Star us on Github!](https://www.github.com/topoteretes/cognee)

Jump into the world of RAG architecture, inspired by human cognitive processes, using Python.
<a href="https://www.cognee.ai" target="_blank">Cognee</a> runs in iterations, from POC towards production ready code.

To read more about the approach and details on cognitive architecture, see the blog post: <a href="https://topoteretes.notion.site/Going-beyond-Langchain-Weaviate-and-towards-a-production-ready-modern-data-platform-7351d77a1eba40aab4394c24bef3a278?pvs=4" target="_blank">AI Applications and RAGs - Cognitive Architecture, Testability, Production Ready Apps</a>  

Try it yourself on Whatsapp with one of our <a href="https://keepi.ai">partners</a> by typing `/save {content you want to save}` followed by `/query {knowledge you saved previously}`



## Getting started

In order to run cognee you need to have <a href="https://docs.docker.com/get-docker" target="_blank">Docker</a> installed on your machine.

Run <a href="https://www.cognee.ai" target="_blank">Cognee</a> in a couple of steps:
- Run  `cp .env.template .env` in your terminal and set all the environment variables
- Run `docker compose up` in order to start graph and relational databases
- Run `docker compose up cognee` in order start Cognee

<!-- Send API requests add-memory, user-query-to-graph, document-to-graph-db, user-query-processor to the locahost:8000 -->

## Debugging
In order to run Cognee with debugger attached you need to build the Cognee image with the `DEBUG` flag set to true.

- `docker compose build cognee --no-cache --build-arg DEBUG=true`
- `docker compose up cognee`


## Demo

[<img src="https://i3.ytimg.com/vi/yjParvJVgPI/maxresdefault.jpg" width="100%">](https://www.youtube.com/watch?v=yjParvJVgPI "Learn about cognee: 55")
## Current Focus

### Integration with keepi.ai and other apps
- Cognee uses Neo4j graph database to map user data into a graph structure consisting of semantic, episodic, and procedural memory.
- Stores data and files through the WhatsApp chatbot <a href="https://keepi.ai">keepi.ai</a>
- Uses the graph to answer user queries and store new information in the graph.


## Architecture

### How Cognee Enhances Your Contextual Memory

Our framework for the OpenAI, Graph (Neo4j) and Vector (Weaviate) databases introduces three key enhancements:

- Query Classifiers: Navigate information graph using Pydantic OpenAI classifiers.
- Document Topology: Structure and store documents in public and private domains.
- Personalized Context: Provide a context object to the LLM for a better response.

</br>

![Image](assets/architecture.png)
