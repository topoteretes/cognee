import os
import asyncio
import cognee

async def main():
    """
    Send cognee traces to Langfuse natively over OpenTelemetry.

    Cognee emits rich OpenTelemetry (OTEL) spans by default. Instead of double-instrumenting
    with a separate Langfuse SDK, you can simply point cognee's existing OTEL exporter
    directly to Langfuse.

    Steps to run this example:
    1. Sign up at https://langfuse.com/ and create a project to get your API keys.
    2. Set your environment variables below (or in your .env file).
    3. Run this script.
    4. Open your Langfuse dashboard and navigate to "Traces".
    """
    print("Setting up Langfuse telemetry...")
    
    # 1. Configure Langfuse credentials
    # Cognee will automatically construct the Basic Auth headers and OTLP endpoint
    # required by Langfuse when these variables are set.
    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..." # Replace with your public key
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..." # Replace with your secret key
    
    # The host defaults to https://cloud.langfuse.com if not specified
    # os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com" 
    
    # 2. Add some data
    print("Adding data...")
    await cognee.add("Cognee turns your unstructured data into a graph memory.")
    
    # 3. Cognify! 
    # Because Langfuse keys are set, cognee will stream the execution traces natively
    # via the OpenTelemetry HTTP exporter. LLM calls will automatically be mapped 
    # to Langfuse 'Generations' with token counts and costs.
    print("Cognifying... (Check your Langfuse dashboard!)")
    await cognee.cognify()
    
    # 4. Search
    print("Searching...")
    results = await cognee.search("What does cognee do?")
    print(results)
    
if __name__ == "__main__":
    asyncio.run(main())
