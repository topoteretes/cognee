from datetime import timedelta
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
import os
import asyncio

# ==============================
#  Production LLM API Integration
# ==============================

class LLMClient:
    """
    A simple production-ready LLM client for Cognee.
    Falls back to mock behavior if no API key is found.
    """

    def __init__(self, provider="openai"):
        self.provider = provider.lower()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key and self.provider == "openai":
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None  # fallback (mock mode)

    def generate(self, prompt: str):
        if not self.client:
            print("⚠️ Running in mock mode: No API key found.")
            return f"[MOCK RESPONSE] {prompt[:100]}..."
        else:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a semantic reasoning assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()


# ==============================
#  MCP Client Logic
# ==============================

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="uv",  # Executable
    args=["--directory", ".", "run", "cognee"],  # Optional command line arguments
    env=None,  # Optional environment variables
)

# Sample text for semantic understanding
text = """
Artificial intelligence, or AI, is technology that enables computers
and machines to simulate human intelligence and problem-solving
capabilities.
On its own or combined with other technologies (e.g., sensors,
geolocation, robotics) AI can perform tasks that would otherwise
require human intelligence or intervention. Digital assistants, GPS
guidance, autonomous vehicles, and generative AI tools (like Open
AI's Chat GPT) are just a few examples of AI in the daily news and
our daily lives.
As a field of computer science, artificial intelligence encompasses
(and is often mentioned together with) machine learning and deep
learning. These disciplines involve the development of AI
algorithms, modeled after the decision-making processes of the human
brain, that can ‘learn’ from available data and make increasingly
more accurate classifications or predictions over time.
"""

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, timedelta(minutes=3)) as session:
            await session.initialize()

            toolResult = await session.list_tools()
            toolResult = await session.call_tool("prune", arguments={})
            toolResult = await session.call_tool("cognify", arguments={})
            toolResult = await session.call_tool(
                "search", arguments={"search_type": "GRAPH_COMPLETION"}
            )

            print(f"\nCognify result: {toolResult.content}")

            # ==============================
            #  Real LLM Integration Section
            # ==============================
            llm = LLMClient()
            response = llm.generate("Summarize this text:\n" + text)
            print("\n🔹 Real LLM Response:\n", response)


if __name__ == "__main__":
    asyncio.run(run())
