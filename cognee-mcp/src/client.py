from datetime import timedelta
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="mcp",  # Executable
    args=["run", "src/server.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)

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

            toolResult = await session.call_tool("cognify", arguments={"text": text})
            # toolResult = await session.call_tool("search", arguments={"search_query": "AI"})

            print(f"Cognify result: {toolResult}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
