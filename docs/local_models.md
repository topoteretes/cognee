# Running cognee with local models

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
cognee.config.llm_provider = 'ollama'

```
You can also set the HOST and model name:

```bash

cognee.config.llm_endpoint = "http://localhost:11434/v1"
cognee.config.llm_model = "mistral:instruct"
```


### Anyscale

```bash
LLM_PROVIDER = 'custom'

```
Otherwise, you can set the configuration for the model:

```bash
cognee.config.llm_provider = 'custom'

```
You can also set the HOST  and model name:
```bash
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
LLM_ENDPOINT = "https://api.endpoints.anyscale.com/v1"
LLM_API_KEY = "your_api_key"
```

You can set the same way HOST and model name for any other provider that has an API endpoint.







