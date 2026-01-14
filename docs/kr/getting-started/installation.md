# Installation

> Set up your environment and install Cognee

Set up your environment and install Cognee to start building AI memory.

<Info>
  Python **3.9 – 3.12** is required to run Cognee.
</Info>

## Prerequisites

<AccordionGroup>
  <Accordion title="Environment Configuration">
    * We recommend creating a `.env` file in your project root
    * Cognee supports many configuration options, and a `.env` file keeps them organized
  </Accordion>

  <Accordion title="API Keys & Models">
    You have two main options for configuring LLM and embedding providers:

    **Option 1: OpenAI (Simplest)**

    * Single API key handles both LLM and embeddings
    * Uses gpt-4o-mini for LLM and text-embedding-3-small for embeddings by default
    * Works out of the box with minimal configuration

    **Option 2: Other Providers**

    * Configure both LLM and embedding providers separately
    * Supports Gemini, Anthropic, Ollama, and more
    * Requires setting both `LLM_*` and `EMBEDDING_*` variables

    <Info>
      By default, Cognee uses OpenAI for both LLMs and embeddings. If you change the LLM provider but don't configure embeddings, it will still default to OpenAI.
    </Info>
  </Accordion>

  <Accordion title="Virtual Environment">
    * We recommend using [uv](https://github.com/astral-sh/uv) for virtual environment management
    * Run the following commands to create and activate a virtual environment:

    ```bash  theme={null}
    uv venv && source .venv/bin/activate
    ```
  </Accordion>

  <Accordion title="Optional">
    <AccordionGroup>
      <Accordion title="Database">
        * PostgreSQL database is required if you plan to use PostgreSQL as your relational database (requires `postgres` extra)
      </Accordion>
    </AccordionGroup>
  </Accordion>
</AccordionGroup>

## Setup

<Tabs>
  <Tab title="OpenAI (Recommended)">
    <Card>
      **Environment:** Add your OpenAI API key to your `.env` file:

      ```bash  theme={null}
      LLM_API_KEY="your_openai_api_key"
      ```

      **Installation:** Install Cognee with all extras:

      ```bash  theme={null}
      uv pip install cognee
      ```

      **What this gives you**: Cognee installed with default local databases (SQLite, LanceDB, Kuzu) — no external servers required.

      <Info>
        This single API key handles both LLM and embeddings. We use gpt-4o-mini for the LLM model and text-embedding-3-small for embeddings by default.
      </Info>
    </Card>
  </Tab>

  <Tab title="Other Providers (Gemini, Anthropic, etc.)">
    <Card>
      **Environment:** Configure both LLM and embedding providers in your `.env` file. Here is an example for Gemini:

      ```bash  theme={null}
      # LLM
      LLM_PROVIDER="gemini"
      LLM_MODEL="gemini/gemini-flash-latest"
      LLM_API_KEY="your_gemini_api_key"

      # Embeddings
      EMBEDDING_PROVIDER="gemini"
      EMBEDDING_MODEL="gemini/text-embedding-004"
      EMBEDDING_API_KEY="your_gemini_api_key"
      ```

      <Info>
        Make sure to configure both LLM and embedding settings. If you only set one, the other will default to OpenAI.
      </Info>

      **Installation:** Install Cognee with provider-specific extras (`gemini`, `anthropic`, `ollama`, `mistral`, `huggingface`, or `groq`) for example:

      ```bash  theme={null}
      uv pip install cognee[gemini]
      ```

      **What this gives you**: Cognee installed with your chosen providers and default local databases.

      For detailed configuration options, see our [LLM](/setup-configuration/llm-providers) and [Embeddings](/setup-configuration/embedding-providers) guides.
    </Card>
  </Tab>
</Tabs>

## Next Steps

<CardGroup cols={2}>
  <Card title="Run Your First Example" href="/getting-started/quickstart" icon="play">
    **Quickstart Tutorial**

    Get started with Cognee by running your first knowledge graph example.
  </Card>

  <Card title="Explore Advanced Features" href="/core-concepts" icon="compass">
    **Core Concepts**

    Dive deeper into Cognee's powerful features and capabilities.
  </Card>
</CardGroup>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt