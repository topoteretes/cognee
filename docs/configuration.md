# Configuration



## 🚀 Configure Vector and Graph Stores

You can configure the vector and graph stores using the environment variables in your .env file or programmatically.
We use [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/#dotenv-env-support)

We have a global configuration object (cognee.config) and individual configurations on pipeline and data store levels

Check available configuration options:
``` python
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.relational import get_relational_config
from cognee.infrastructure.llm.config import get_llm_config
print(get_vectordb_config().to_dict())
print(get_graph_config().to_dict())
print(get_relational_config().to_dict())
print(get_llm_config().to_dict())

```

Setting the environment variables in your .env file, and Pydantic will pick them up:

```bash
GRAPH_DATABASE_PROVIDER = 'lancedb'

```
Otherwise, you can set the configuration yourself:

```python
cognee.config.set_llm_provider('ollama')
```

## 🚀 Getting Started with Local Models

You'll need to run the local model on your machine or use one of the providers hosting the model.
!!! note "We had some success with mixtral, but 7b models did not work well. We recommend using mixtral for now."

### Ollama 

Set up Ollama by following instructions on [Ollama website](https://ollama.com/)


Set the environment variable in your .env to use the model

```bash
LLM_PROVIDER = 'ollama'

```
Otherwise, you can set the configuration for the model:

```bash
cognee.config.set_llm_provider('ollama')

```
You can also set the HOST and model name:

```bash
cognee.config.set_llm_endpoint("http://localhost:11434/v1")
cognee.config.set_llm_model("mistral:instruct")
```


### Anyscale

```bash
LLM_PROVIDER = 'custom'

```
Otherwise, you can set the configuration for the model:

```bash
cognee.config.set_llm_provider('custom')

```
You can also set the HOST  and model name:
```bash
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
LLM_ENDPOINT = "https://api.endpoints.anyscale.com/v1"
LLM_API_KEY = "your_api_key"
```

You can set the same way HOST and model name for any other provider that has an API endpoint.







