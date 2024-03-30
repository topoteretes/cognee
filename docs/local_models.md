# Running cognee with local models

## 1. 🚀 Getting Started with Local Models

You'll need to run the local model on your machine or use one of the providers hosting the model.

### Ollama 

Download the model from the [Ollama website](https://ollama.com/)


Set the environment variable to use the model

```bash
LLM_PROVIDER = 'ollama'

```
You can also set the HOST and model name

CUSTOM_OLLAMA_ENDPOINT= "http://localhost:11434/v1"
CUSTOM_OLLAMA_MODEL = "mistral:instruct"


### Anyscale

```bash
LLM_PROVIDER = 'custom'

```
You can also set the HOST  and model name
CUSTOM_LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
CUSTOM_ENDPOINT = "https://api.endpoints.anyscale.com/v1"
CUSTOM_LLM_API_KEY = "your_api_key"


You can also set the HOST and model name for any other provider






