"""Send cognee traces to Langfuse natively over OpenTelemetry.

Cognee already emits rich OpenTelemetry (OTEL) spans. Instead of double-instrumenting
with a separate Langfuse SDK, you point cognee's existing OTLP exporter at Langfuse —
Langfuse is just another OTLP destination, like Dash0 or Datadog.

To run:
  1. Create a Langfuse project (https://langfuse.com) to get your API keys.
  2. Export the keys BEFORE running (so cognee's config picks them up). cognee builds
     the OTLP endpoint + Basic-auth header and turns tracing on automatically:

       export LANGFUSE_PUBLIC_KEY="pk-lf-..."
       export LANGFUSE_SECRET_KEY="sk-lf-..."
       # optional; defaults to https://cloud.langfuse.com
       export LANGFUSE_HOST="https://us.cloud.langfuse.com"

  3. python examples/guides/langfuse_telemetry.py
  4. Open your Langfuse dashboard -> "Traces". LLM calls appear as Generations.
"""

import os
import asyncio

import cognee
from cognee import SearchType


async def main():
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        raise SystemExit(
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (see this file's docstring)."
        )

    # Because the Langfuse keys are set, cognee streams execution traces to Langfuse
    # over the existing OTLP HTTP exporter; LLM calls render as Generations.
    print("Remembering... (check your Langfuse dashboard)")
    await cognee.remember(
        "Cognee turns your unstructured data into a graph memory.", self_improvement=False
    )

    print("Recalling...")
    results = await cognee.recall("What does cognee do?", query_type=SearchType.GRAPH_COMPLETION)
    print([result.text for result in results])


if __name__ == "__main__":
    asyncio.run(main())
