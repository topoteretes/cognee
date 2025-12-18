# Personal Assistant with Memory

An executive assistant for travel planning that remembers your preferences and plans itineraries accordingly using **Cognee** (Semantic memory system) and **Agno** (Agent framework).

## Overview

This example demonstrates how to build a personal assistant that:
- **Remembers** your travel preferences and dietary restrictions
- **Plans** personalized itineraries based on stored preferences
- **Searches** memory to retrieve relevant information when making recommendations
- **Adapts** recommendations based on your specific needs (e.g., vegetarian meals, hotel preferences, beach locations)

The assistant uses Cognee's Semantic Memory layer to store and retrieve user preferences, enabling context-aware travel planning.

## Architecture

- **Agent Framework**: [Agno](https://github.com/agno-agi/agno/) - For building intelligent agents
- **Memory Layer**: [Cognee](https://github.com/topoteretes/cognee/) - Semantic knowledge graph for persistent memory
- **Two Implementation Options**:
  - `app.py`: Uses OpenAI (default stack)
  - `custom_stack.py`: Uses Gemini + Qdrant + FastEmbed (custom stack)

## Prerequisites

- Python >= 3.10 and < 3.14
- [uv](https://github.com/astral-sh/uv) package manager
- API keys for your chosen LLM provider (OpenAI or Google)

## Installation

### 1. Create Virtual Environment

```bash
uv venv
source .venv/bin/activate
```

### 2. Install Dependencies

Choose one of the following based on which implementation you want to use:

#### Option A: Default Stack (`app.py`)

Uses OpenAI for both agent execution and memory layer.

```bash
uv add cognee agno
```

#### Option B: Custom Stack (`custom_stack.py`)

Uses Gemini (LLM), Qdrant (vector store), and FastEmbed (embeddings).

```bash
uv add google-genai cognee-community-vector-adapter-qdrant fastembed
```

## Configuration

Create a `.env` file in the project root with the appropriate environment variables.

### For `app.py` (OpenAI Stack)

```env
# OpenAI API key for Agent execution
OPENAI_API_KEY=<your-openai-api-key>

# OpenAI API key for Cognee memory layer
LLM_API_KEY=<your-openai-api-key>
```

### For `custom_stack.py` (Gemini + Qdrant + FastEmbed)

```env
# Google API key for agent execution
GOOGLE_API_KEY=<your-google-api-key>

# Cognee LLM configuration
LLM_PROVIDER=gemini
LLM_MODEL=gemini/gemini-2.5-flash
LLM_API_KEY=<your-google-api-key>

# Cognee embedding configuration
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=jinaai/jina-embeddings-v2-base-en
EMBEDDING_DIMENSIONS=768

# Qdrant vector store configuration
QDRANT_URL=<your-qdrant-cluster-endpoint>
QDRANT_API_KEY=<your-qdrant-api-key>
```

**Stack Components:**
- **Vector Store**: Qdrant
- **LLM**: Gemini 2.5 Flash
- **Embedding**: FastEmbed (Jina embeddings v2)

## Usage

### Running the Default Stack (`app.py`)

```bash
python app.py
```

The agent will:
1. Store your preferences in Cognee's memory
2. Demonstrate retrieval by planning a restaurant itinerary for Rome

### Running the Custom Stack (`custom_stack.py`)

```bash
uv run custom_stack.py
```

Same functionality as above, but using the custom stack configuration.

## How It Works

1. **Memory Storage**: When you provide preferences, the agent uses `add_memory` to store them in Cognee's semantic knowledge graph
2. **Memory Retrieval**: When planning itineraries, the agent uses `search_memory` to find relevant preferences
3. **Context-Aware Planning**: The agent applies retrieved preferences to generate personalized recommendations

### Example Interaction

```python
# Store preferences
agent.print_response(MY_PREFERENCE, stream=True)

# Get recommendations based on stored preferences
agent.print_response("I am visiting Rome, give me restaurants list to stop by", stream=True)
```
## Project Structure

```
agno-agents/
├── app.py              # Default implementation (OpenAI)
├── custom_stack.py     # Custom implementation (Gemini + Qdrant)
├── tools.py            # CogneeTools integration for Agno
├── constants.py        # Agent instructions and user preferences
├── README.md           # This file
└── .env                # Environment variables (create this)
```

## Customization

You can customize the assistant by modifying:

- **`constants.py`**: Update `MY_PREFERENCE` with your own preferences and `INSTRUCTIONS` to change agent behavior
- **`tools.py`**: Extend `CogneeTools` to add more memory operations
- **Environment variables**: Switch between different LLM providers, vector stores, or embedding models

## Troubleshooting

- **Import errors**: Ensure all dependencies are installed with `uv sync`
- **API key errors**: Verify your `.env` file contains valid API keys
- **Qdrant connection**: For `custom_stack.py`, ensure your Qdrant cluster is accessible and credentials are correct