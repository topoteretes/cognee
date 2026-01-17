# LLM Providers

> Configure LLM providers for text generation and reasoning in Cognee

LLM (Large Language Model) providers handle text generation, reasoning, and structured output tasks in Cognee. You can choose from cloud providers like OpenAI and Anthropic, or run models locally with Ollama.

<Info>
  **New to configuration?**

  See the [Setup Configuration Overview](./overview) for the complete workflow:

  install extras → create `.env` → choose providers → handle pruning.
</Info>

## Supported Providers

Cognee supports multiple LLM providers:

* **OpenAI** — GPT models via OpenAI API (default)
* **Azure OpenAI** — GPT models via Azure OpenAI Service
* **Google Gemini** — Gemini models via Google AI
* **Anthropic** — Claude models via Anthropic API
* **AWS Bedrock** — Models available via AWS Bedrock
* **Ollama** — Local models via Ollama
* **LM Studio** — Local models via LM Studio
* **Custom** — OpenAI-compatible endpoints (like vLLM)

<Warning>
  **LLM/Embedding Configuration**: If you configure only LLM or only embeddings, the other defaults to OpenAI. Ensure you have a working OpenAI API key, or configure both LLM and embeddings to avoid unexpected defaults.
</Warning>

## Configuration

<Accordion title="Environment Variables">
  Set these environment variables in your `.env` file:

  * `LLM_PROVIDER` — The provider to use (openai, gemini, anthropic, ollama, custom)
  * `LLM_MODEL` — The specific model to use
  * `LLM_API_KEY` — Your API key for the provider
  * `LLM_ENDPOINT` — Custom endpoint URL (for Azure, Ollama, or custom providers)
  * `LLM_API_VERSION` — API version (for Azure OpenAI)
  * `LLM_MAX_TOKENS` — Maximum tokens per request (optional)
</Accordion>

## Provider Setup Guides

<AccordionGroup>
  <Accordion title="OpenAI (Default)">
    OpenAI is the default provider and works out of the box with minimal configuration.

    ```dotenv  theme={null}
    LLM_PROVIDER="openai"
    LLM_MODEL="gpt-4o-mini"
    LLM_API_KEY="sk-..."
    # Optional overrides
    # LLM_ENDPOINT=https://api.openai.com/v1
    # LLM_API_VERSION=
    # LLM_MAX_TOKENS=16384
    ```
  </Accordion>

  <Accordion title="Azure OpenAI">
    Use Azure OpenAI Service with your own deployment.

    ```dotenv  theme={null}
    LLM_PROVIDER="openai"
    LLM_MODEL="azure/gpt-4o-mini"
    LLM_ENDPOINT="https://<your-resource>.openai.azure.com/openai/deployments/gpt-4o-mini"
    LLM_API_KEY="az-..."
    LLM_API_VERSION="2024-12-01-preview"
    ```
  </Accordion>

  <Accordion title="Google Gemini">
    Use Google's Gemini models for text generation.

    ```dotenv  theme={null}
    LLM_PROVIDER="gemini"
    LLM_MODEL="gemini/gemini-2.0-flash"
    LLM_API_KEY="AIza..."
    # Optional
    # LLM_ENDPOINT=https://generativelanguage.googleapis.com/
    # LLM_API_VERSION=v1beta
    ```
  </Accordion>

  <Accordion title="Anthropic">
    Use Anthropic's Claude models for reasoning tasks.

    ```dotenv  theme={null}
    LLM_PROVIDER="anthropic"
    LLM_MODEL="claude-3-5-sonnet-20241022"
    LLM_API_KEY="sk-ant-..."
    ```
  </Accordion>

  <Accordion title="AWS Bedrock">
    Use models available on AWS Bedrock for various tasks. For Bedrock specifically, you will need to
    also specify some information regarding AWS.

    ```dotenv  theme={null}
    LLM_API_KEY="<your_bedrock_api_key>"
    LLM_MODEL="eu.amazon.nova-lite-v1:0"
    LLM_PROVIDER="bedrock"
    LLM_MAX_TOKENS="16384"
    AWS_REGION="<your_aws_region>"
    AWS_ACCESS_KEY_ID="<your_aws_access_key_id>"
    AWS_SECRET_ACCESS_KEY="<your_aws_secret_access_key>"
    AWS_SESSION_TOKEN="<your_aws_session_token>"

    # Optional parameters
    #AWS_BEDROCK_RUNTIME_ENDPOINT="bedrock-runtime.eu-west-1.amazonaws.com"
    #AWS_PROFILE_NAME="<path_to_your_aws_credentials_file>"
    ```

    There are **multiple ways of connecting** to Bedrock models:

    1. Using an API key and region. Simply generate you key on AWS, and put it in the `LLM_API_KEY` env variable.
    2. Using AWS Credentials. You can only specify `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, no need for the `LLM_API_KEY`.
       In this case, if you are using temporary credentials (e.g. `AWS_ACCESS_KEY_ID` starting with `ASIA...`), then you also
       must specify the `AWS_SESSION_TOKEN`.
    3. Using AWS profiles. Create a file called something like `/.aws/credentials`, and store your credentials inside it.

    **Installation**: Install the required dependency:

    ```bash  theme={null}
    pip install cognee[aws]
    ```

    <Info>
      **Model Name**
      The name of the model might differ based on the region (the name begins with **eu** for Europe, **us** of USA, etc.)
    </Info>
  </Accordion>

  <Accordion title="Ollama (Local)">
    Run models locally with Ollama for privacy and cost control.

    ```dotenv  theme={null}
    LLM_PROVIDER="ollama"
    LLM_MODEL="llama3.1:8b"
    LLM_ENDPOINT="http://localhost:11434/v1"
    LLM_API_KEY="ollama"
    ```

    **Installation**: Install Ollama from [ollama.ai](https://ollama.ai) and pull your desired model:

    ```bash  theme={null}
    ollama pull llama3.1:8b
    ```

    ### Known Issues

    * **Requires `HUGGINGFACE_TOKENIZER`**: Ollama currently needs this env var set even when used only as LLM. Fix in progress.
    * **`NoDataError` with mixed providers**: Using Ollama as LLM and OpenAI as embedding provider may fail with `NoDataError`. Workaround: use the same provider for both.
  </Accordion>

  <Accordion title="LM Studio (Local)">
    Run models locally with LM Studio for privacy and cost control.

    ```dotenv  theme={null}
    LLM_PROVIDER="custom"
    LLM_MODEL="lm_studio/magistral-small-2509"
    LLM_ENDPOINT="http://127.0.0.1:1234/v1"
    LLM_API_KEY="."
    LLM_INSTRUCTOR_MODE="json_schema_mode"
    ```

    **Installation**: Install LM Studio from [lmstudio.ai](https://lmstudio.ai/) and download your desired model from
    LM Studio's interface.
    Load your model, start the LM Studio server, and Cognee will be able to connect to it.

    <Info>
      **Set up instructor mode**
      The `LLM_INSTRUCTOR_MODE` env variable controls the LiteLLM instructor [mode](https://python.useinstructor.com/modes-comparison/),
      i.e. the model's response type.
      This may vary depending on the model, and you would need to change it accordingly.
    </Info>
  </Accordion>

  <Accordion title="Custom Providers">
    Use OpenAI-compatible endpoints like OpenRouter or other services.

    ```dotenv  theme={null}
    LLM_PROVIDER="custom"
    LLM_MODEL="openrouter/google/gemini-2.0-flash-lite-preview-02-05:free"
    LLM_ENDPOINT="https://openrouter.ai/api/v1"
    LLM_API_KEY="or-..."
    # Optional fallback chain
    # FALLBACK_MODEL=
    # FALLBACK_ENDPOINT=
    # FALLBACK_API_KEY=
    ```

    **Custom Provider Prefixes**: When using `LLM_PROVIDER="custom"`, you must include the correct provider prefix in your model name. Cognee forwards requests to [LiteLLM](https://docs.litellm.ai/docs/providers), which uses these prefixes to route requests correctly.

    Common prefixes include:

    * `hosted_vllm/` — vLLM servers
    * `openrouter/` — OpenRouter
    * `lm_studio/` — LM Studio
    * `openai/` — OpenAI-compatible APIs

    See the [LiteLLM providers documentation](https://docs.litellm.ai/docs/providers) for the full list of supported prefixes.

    Below is an example for vLLm:

    <Accordion title="vLLM">
      Use vLLM for high-performance model serving with OpenAI-compatible API.

      ```dotenv  theme={null}
      LLM_PROVIDER="custom"
      LLM_MODEL="hosted_vllm/<your-model-name>"
      LLM_ENDPOINT="https://your-vllm-endpoint/v1"
      LLM_API_KEY="."
      ```

      **Example with Gemma:**

      ```dotenv  theme={null}
      LLM_PROVIDER="custom"
      LLM_MODEL="hosted_vllm/gemma-3-12b"
      LLM_ENDPOINT="https://your-vllm-endpoint/v1"
      LLM_API_KEY="."
      ```

      <Warning>
        **Important**: The `hosted_vllm/` prefix is required for LiteLLM to correctly route requests to your vLLM server. The model name after the prefix should match the model ID returned by your vLLM server's `/v1/models` endpoint.
      </Warning>

      To find the correct model name, see [their documentation](https://docs.litellm.ai/docs/providers/vllm).
    </Accordion>
  </Accordion>
</AccordionGroup>

## Advanced Options

<Accordion title="Rate Limiting">
  Control client-side throttling for LLM calls to manage API usage and costs.

  **Configuration (in .env):**

  ```dotenv  theme={null}
  LLM_RATE_LIMIT_ENABLED="true"
  LLM_RATE_LIMIT_REQUESTS="60"
  LLM_RATE_LIMIT_INTERVAL="60"
  ```

  **How it works:**

  * **Client-side limiter**: Cognee paces outbound LLM calls before they reach the provider
  * **Moving window**: Spreads allowance across the time window for smoother throughput
  * **Per-process scope**: In-memory limits don't share across multiple processes/containers
  * **Auto-applied**: Works with all providers (OpenAI, Gemini, Anthropic, Ollama, Custom)

  **Example**: `60` requests per `60` seconds ≈ 1 request/second average rate.
</Accordion>

## Notes

* If `EMBEDDING_API_KEY` is not set, Cognee falls back to `LLM_API_KEY` for embeddings
* Rate limiting helps manage API usage and costs
* Structured output frameworks ensure consistent data extraction from LLM responses
* If you are using `Instructor` as the structured output framework, you can control the
  response type of the LLM through the `LLM_INSTRUCTOR_MODE` env variable, which sets the
  corresponding instructor [mode](https://python.useinstructor.com/modes-comparison/)
  (e.g. `json_mode` for JSON output)

<Columns cols={3}>
  <Card title="Embedding Providers" icon="layers" href="/setup-configuration/embedding-providers">
    Configure embedding providers for semantic search
  </Card>

  <Card title="Overview" icon="settings" href="/setup-configuration/overview">
    Return to setup configuration overview
  </Card>

  <Card title="Relational Databases" icon="database" href="/setup-configuration/relational-databases">
    Set up SQLite or Postgres for metadata storage
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt