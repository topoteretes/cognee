# Cognee

AI Applications and RAGs - Cognitive Architecture, Testability, Production Ready Apps



<p align="left">
  <a href="https://prometh.ai//#gh-light-mode-only">
    <img src="assets/topoteretes_logo.png" width="5%" alt="promethAI logo" />
  </a>

  
</p>

<p align="left"><i>Open-source framework for building and testing RAGs and Cognitive Architectures, designed for accuracy, transparency, and control.</i></p>

<p align="left">
<a href="https://github.com/topoteretes/cognee/fork" target="blank">
<img src="https://img.shields.io/github/forks/topoteretes/cognee?style=for-the-badge" alt="cognee forks"/>
</a>

<a href="https://github.com/topoteretes/cognee/stargazers" target="blank">
<img src="https://img.shields.io/github/stars/topoteretes/cognee?style=for-the-badge" alt="cognee stars"/>
</a>
<a href="https://github.com/topoteretes/cognee/pulls" target="blank">
<img src="https://img.shields.io/github/issues-pr/topoteretes/cognee?style=for-the-badge" alt="cognee pull-requests"/>
</a>
<a href='https://github.com/topoteretes/cognee/releases'>
<img src='https://img.shields.io/github/release/topoteretes/cognee?&label=Latest&style=for-the-badge'>
</a>

</p>

[//]: # (<p align="center"><b>Follow PromethAI </b></p>)

[//]: # (<p align="center">)

[//]: # (<a href="https://twitter.com/_promethAI" target="blank">)

[//]: # (<img src="https://img.shields.io/twitter/follow/_promethAI?label=Follow: _promethAI&style=social" alt="Follow _promethAI"/>)

[//]: # (</a>)

[//]: # (<p align="center">)

[//]: # (<a href="https://prometh.ai" target="_blank"><img src="https://img.shields.io/twitter/url?label=promethAI Website&logo=website&style=social&url=https://github.com/topoteretes/PromethAI-Memory"/></a>)

[//]: # (<p align="center">)

[//]: # (<a href="https://www.youtube.com/@_promethAI" target="_blank"><img src="https://img.shields.io/twitter/url?label=Youtube&logo=youtube&style=social&url=https://github.com/topoteretes/PromethAI-Memory"/></a>)

[//]: # (</p>)


<p align="left"><b>Share promethAI Repository</b></p>

<p align="left">

<a href="https://twitter.com/intent/tweet?text=Check%20this%20GitHub%20repository%20out.%20promethAI%20-%20Let%27s%20you%20easily%20build,%20manage%20and%20run%20useful%20autonomous%20AI%20agents.&url=https://github.com/topoteretes/PromethAI-Backend-Backend&hashtags=promethAI,AGI,Autonomics,future" target="blank">
<img src="https://img.shields.io/twitter/follow/_promethAI?label=Share Repo on Twitter&style=social" alt="Follow _promethAI"/></a> 
<a href="https://t.me/share/url?text=Check%20this%20GitHub%20repository%20out.%20promethAI%20-%20Let%27s%20you%20easily%20build,%20manage%20and%20run%20useful%20autonomous%20AI%20agents.&url=https://github.com/topoteretes/PromethAI-Backend" target="_blank"><img src="https://img.shields.io/twitter/url?label=Telegram&logo=Telegram&style=social&url=https://github.com/topoteretes/PromethAI-Backend" alt="Share on Telegram"/></a>
<a href="https://api.whatsapp.com/send?text=Check%20this%20GitHub%20repository%20out.%20promethAI%20-%20Let's%20you%20easily%20build,%20manage%20and%20run%20useful%20autonomous%20AI%20agents.%20https://github.com/topoteretes/PromethAI-Backend"><img src="https://img.shields.io/twitter/url?label=whatsapp&logo=whatsapp&style=social&url=https://github.com/topoteretes/PromethAI-Backend" /></a> <a href="https://www.reddit.com/submit?url=https://github.com/topoteretes/PromethAI-Backend&title=Check%20this%20GitHub%20repository%20out.%20promethAI%20-%20Let's%20you%20easily%20build,%20manage%20and%20run%20useful%20autonomous%20AI%20agents.
" target="blank">
<img src="https://img.shields.io/twitter/url?label=Reddit&logo=Reddit&style=social&url=https://github.com/topoteretes/PromethAI-Backend" alt="Share on Reddit"/>
</a> <a href="mailto:?subject=Check%20this%20GitHub%20repository%20out.&body=promethAI%20-%20Let%27s%20you%20easily%20build,%20manage%20and%20run%20useful%20autonomous%20AI%20agents.%3A%0Ahttps://github.com/topoteretes/PromethAI-Backend" target="_blank"><img src="https://img.shields.io/twitter/url?label=Gmail&logo=Gmail&style=social&url=https://github.com/topoteretes/PromethAI-Backend"/></a> <a href="https://www.buymeacoffee.com/promethAI" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="23" width="100" style="border-radius:1px"></a>

</p>

<hr>





This repo is built to test and evolve RAG architecture, inspired by human cognitive processes, using Python. 
It's aims to be production ready, testable, and give great visibility in how we build RAG applications.
It runs in iterations, from POC towards production ready code.
To read more about the approach and details on cognitive architecture, see the blog post: [AI Applications and RAGs - Cognitive Architecture, Testability, Production Ready Apps](https://topoteretes.notion.site/Going-beyond-Langchain-Weaviate-and-towards-a-production-ready-modern-data-platform-7351d77a1eba40aab4394c24bef3a278?pvs=4)
Try it on Whatsapp with one of our partners Keepi.ai by typing /save {content} followed by /query {content}


### Current Focus

#### Level 5 - Integration to keepi.ai and other apps
Scope: Use Neo4j to map user preferences into a graph structure consisting of semantic, episodic, and procedural memory. 
Fetch information and store information and files on Whatsapp chatbot using Keepi.ai
Use the graph to answer user queries and store new information in the graph.



![Image](https://github.com/topoteretes/PromethAI-Memory/blob/main/level_4/User_graph.png)


### Installation

### Run cognee

Make sure you have Docker, Poetry, and Python 3.11 installed and postgres installed.

Copy the .env.example to .env and fill in the variables

``` poetry shell ```

```docker compose up   ```

And send API requests add-memory, user-query-to-graph, document-to-graph-db, user-query-processor to the locahost:8000


If you are running natively, change ENVIRONMENT to local in the .env file
If you are running in docker, change ENVIRONMENT to postgres in the .env file













