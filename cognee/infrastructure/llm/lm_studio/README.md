# LM Studio Adapter for Cognee

This adapter integrates LM Studio with Cognee, allowing you to use local LLMs through LM Studio's OpenAI-compatible API.

## Features

- Full support for LM Studio's OpenAI-compatible API
- Structured output generation using instructor
- Image analysis with multimodal models
- Embeddings generation
- Fallback tokenization when tiktoken is not available

## Setup

1. Install and run LM Studio from [lmstudio.ai](https://lmstudio.ai/)
2. Start the LM Studio API server (in LM Studio, go to the API tab and click "Start Server")
3. Configure Cognee to use LM Studio as the LLM provider

## Configuration

Add the following to your `.env` file or environment variables:

```
LLM_PROVIDER=lm_studio
LLM_ENDPOINT=http://localhost:1234/v1
LLM_API_KEY=lm-studio  # Can be any string
LLM_MODEL=<model-name>  # The name of the model loaded in LM Studio
```

## Usage Examples

### Basic Usage

```python
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.shared.data_models import SummarizedContent

# Get the configured LLM client (LM Studio in this case)
llm_client = get_llm_client()

# Generate a structured output
system_prompt = "Summarize the following text."
text_input = "LM Studio is a powerful tool for running local LLMs..."
result = await llm_client.acreate_structured_output(
    text_input, 
    system_prompt, 
    SummarizedContent
)
```

### Image Analysis

```python
llm_client = get_llm_client()

# Analyze an image (requires a multimodal model)
image_description = llm_client.transcribe_image(
    "path/to/image.jpg", 
    prompt="Describe what you see in this image in detail."
)
```

### Embeddings

```python
llm_client = get_llm_client()

# Generate embeddings (requires an embedding model)
embeddings = llm_client.create_embeddings("Text to embed")
```

## Supported Parameters

The LM Studio adapter supports the following parameters:

- `model`: The model to use
- `temperature`: Controls randomness (0-1)
- `max_tokens`: Maximum number of tokens to generate
- `top_p`: Controls diversity via nucleus sampling
- `top_k`: Controls diversity via top-k sampling
- `presence_penalty`: Penalizes repeated tokens
- `frequency_penalty`: Penalizes frequent tokens
- `repeat_penalty`: Penalizes repetition (specific to LM Studio)

## Troubleshooting

- **Connection Issues**: Ensure the LM Studio API server is running and accessible at the configured endpoint
- **Model Not Found**: Make sure the model is loaded in LM Studio before making API calls
- **Multimodal Support**: For image analysis, ensure you're using a model with vision capabilities
- **Embeddings**: For embeddings, use a dedicated embedding model or a model that supports embeddings

## Dependencies

- `instructor`: For structured output generation
- `openai`: For API communication
- `tiktoken`: For tokenization (optional, falls back to a simple tokenizer if not available)
