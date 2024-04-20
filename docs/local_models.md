# Running cognee with local models

## 🚀 Getting Started with Local Models

You'll need to run the local model on your machine or use one of the providers hosting the model.
!!! note "We had some success with mixtral, but 7b models did not work well. We recommend using mixtral for now."

### Ollama 

Set up Ollama by following instructions on [Ollama website](https://ollama.com/)


Set the environment variable to use the model

```bash
LLM_PROVIDER = 'ollama'

```
Otherwise, you can set the configuration for the model:

```bash
from cognee.infrastructure import infrastructure_config
infrastructure_config.set_config({
    "llm_provider": 'ollama'
})

```
You can also set the HOST and model name:

```bash

CUSTOM_OLLAMA_ENDPOINT= "http://localhost:11434/v1"
CUSTOM_OLLAMA_MODEL = "mistral:instruct"
```


### Anyscale

```bash
LLM_PROVIDER = 'custom'

```
Otherwise, you can set the configuration for the model:

```bash
from cognee.infrastructure import infrastructure_config
infrastructure_config.set_config({
    "llm_provider": 'custom'
})

```
You can also set the HOST  and model name:
```bash
CUSTOM_LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
CUSTOM_ENDPOINT = "https://api.endpoints.anyscale.com/v1"
CUSTOM_LLM_API_KEY = "your_api_key"
```

You can set the same way HOST and model name for any other provider that has an API endpoint.







