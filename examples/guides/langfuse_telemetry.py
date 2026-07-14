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


async def main():
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        raise SystemExit(
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (see this file's docstring)."
        )

    print("Adding data...")
    await cognee.add("Cognee turns your unstructured data into a graph memory.")

    # Because the Langfuse keys are set, cognee streams execution traces to Langfuse
    # over the existing OTLP HTTP exporter; LLM calls render as Generations.
    print("Cognifying... (check your Langfuse dashboard)")
    await cognee.cognify()

    print("Searching...")
    print(await cognee.search("What does cognee do?"))


if __name__ == "__main__":
    asyncio.run(main())
