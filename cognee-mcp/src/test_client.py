"""Minimal smoke test for the Cognee MCP server.

Run with:

```bash
python src/test_client.py --api-url http://localhost:8000
```
"""

import argparse
import asyncio
import sys
from typing import Optional
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main(api_url: str) -> None:
    python_cmd = sys.executable or "python"
    project_root = Path(__file__).resolve().parents[1]
    server_script = project_root / "src" / "server.py"
    server_params = StdioServerParameters(
        command=python_cmd,
        args=[str(server_script), "--transport", "stdio", "--api-url", api_url],
        cwd=str(project_root),
        env={"LOG_LEVEL": "error"},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, timedelta(minutes=1)) as session:
            await session.initialize()

            tool_names = {tool.name for tool in (await session.list_tools()).tools}
            assert {"list_datasets", "search", "get_dataset_summary"}.issubset(tool_names), (
                tool_names
            )

            datasets = await session.call_tool("list_datasets", {})
            dataset_text = datasets.content[0].text
            print("Datasets:\n", dataset_text)

            first_dataset_id: Optional[str] = None
            for line in dataset_text.splitlines():
                stripped = line.strip()
                if stripped.startswith("id:"):
                    first_dataset_id = stripped.split("id:", 1)[1].strip()
                    if first_dataset_id:
                        break

            if first_dataset_id:
                summary = await session.call_tool(
                    "get_dataset_summary", {"dataset_id": first_dataset_id, "top_k": 1}
                )
                print("\nDataset summary:\n", summary.content[0].text)
            else:
                print("\nNo dataset id detected; skipping summary test")

            search_payload = {
                "query": "Summarize the main takeaways from the onboarding guide",
                "top_k": 3,
                "dataset_ids": [first_dataset_id] if first_dataset_id else None,
                "use_combined_context": True,
            }
            # Remove None entry if no dataset id was found
            search_payload = {k: v for k, v in search_payload.items() if v is not None}

            search_result = await session.call_tool("search", search_payload)
            print("\nSearch result:\n", search_result.content[0].text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-url", required=True, help="Cognee API URL (e.g. http://localhost:8000)"
    )
    args = parser.parse_args()

    asyncio.run(main(args.api_url))
