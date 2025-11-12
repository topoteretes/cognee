"""Search-focused MCP tools for Cognee."""

import json
import re
from typing import List, Literal, Optional

import mcp.types as types
from mcp.server import FastMCP

from cognee.shared.logging_utils import get_logger

from dependencies import DependencyContainer

logger = get_logger()

SearchTypeLiteral = Literal["GRAPH_COMPLETION", "CHUNKS", "SUMMARIES"]
VALID_SEARCH_TYPES = {"GRAPH_COMPLETION", "CHUNKS", "SUMMARIES"}
UUID_PATTERN = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE
)
MAX_SYSTEM_PROMPT_LENGTH = 2000


def setup_tools(mcp: FastMCP, container: DependencyContainer) -> None:
    """Register MCP tools on the provided FastMCP instance."""

    @mcp.tool(
        name="list_datasets",
        description="List knowledge bases with metadata (id, owner, timestamps)",
    )
    async def list_datasets() -> List[types.TextContent]:
        datasets = await container.cognee_client.list_datasets()

        if not datasets:
            message = "No datasets are currently available. Use the Cognee UI to ingest content."
            return [types.TextContent(type="text", text=message)]

        lines = [f"📂 Knowledge Bases ({len(datasets)} total)", ""]
        for index, dataset in enumerate(datasets, 1):
            name = dataset.get("name", "Unnamed dataset")
            identifier = dataset.get("id", "unknown")
            created_at = dataset.get("created_at", "N/A")
            updated_at = dataset.get("updated_at") or "N/A"
            owner = dataset.get("owner_id", "unknown")

            lines.append(f"{index}. {name}")
            lines.append(f"   id: {identifier}")
            lines.append(f"   owner: {owner}")
            lines.append(f"   created: {created_at}")
            lines.append(f"   updated: {updated_at}")
            lines.append("   tip: call get_dataset_summary with the id before searching")
            lines.append("")

        return [types.TextContent(type="text", text="\n".join(lines).strip())]

    @mcp.tool(
        name="search",
        description=(
            "Query Cognee datasets with natural language. Supports combined context,"
            " context-only responses, and node filters."
        ),
    )
    async def search(
        query: str,
        datasets: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        search_type: SearchTypeLiteral = "GRAPH_COMPLETION",
        top_k: int = 10,
        system_prompt: Optional[str] = None,
        use_combined_context: bool = False,
        only_context: bool = False,
        node_name: Optional[List[str]] = None,
    ) -> List[types.TextContent]:
        # Validate query
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        # Validate search_type
        if search_type not in VALID_SEARCH_TYPES:
            raise ValueError(f"search_type must be one of {VALID_SEARCH_TYPES}")

        # Validate top_k
        if top_k < 1 or top_k > 50:
            raise ValueError("top_k must be between 1 and 50")

        # Validate dataset_ids (UUID format)
        if dataset_ids:
            for ds_id in dataset_ids:
                if not ds_id or not ds_id.strip():
                    raise ValueError("dataset_ids must contain non-empty strings")
                if not UUID_PATTERN.match(ds_id):
                    raise ValueError(f"Invalid UUID format for dataset_id: {ds_id}")

        # Validate system_prompt length
        if system_prompt and len(system_prompt) > MAX_SYSTEM_PROMPT_LENGTH:
            raise ValueError(
                f"system_prompt exceeds maximum length of {MAX_SYSTEM_PROMPT_LENGTH} characters"
            )

        # Validate node_name
        if node_name:
            for name in node_name:
                if not name or not name.strip():
                    raise ValueError("node_name must contain non-empty strings")

        payload = await container.cognee_client.search(
            query_text=query,
            datasets=datasets,
            dataset_ids=dataset_ids,
            search_type=search_type,
            top_k=top_k,
            system_prompt=system_prompt,
            use_combined_context=use_combined_context,
            only_context=only_context,
            node_name=node_name,
        )

        formatted = _format_search_payload(payload)
        logger.debug("search completed with %s bytes of payload", len(formatted))
        return [types.TextContent(type="text", text=formatted)]

    @mcp.tool(
        name="get_dataset_summary",
        description="Return the top SUMMARIES entries for a dataset to help choose the right KB",
    )
    async def get_dataset_summary(dataset_id: str, top_k: int = 1) -> List[types.TextContent]:
        if not dataset_id or not dataset_id.strip():
            raise ValueError("dataset_id must be provided")

        if top_k < 1 or top_k > 5:
            raise ValueError("top_k must be between 1 and 5 for summaries")

        payload = await container.cognee_client.search(
            query_text="Provide a concise summary of this dataset's contents and focus areas.",
            datasets=None,
            dataset_ids=[dataset_id],
            search_type="SUMMARIES",
            top_k=top_k,
            system_prompt=None,
            use_combined_context=False,
            only_context=False,
            node_name=None,
        )

        formatted = _format_search_payload(payload)
        text = "Dataset summaries (use them to confirm scope before searching):\n" + formatted
        return [types.TextContent(type="text", text=text)]


def _format_search_payload(payload) -> str:
    """Render API results into a text block for LLM consumption."""

    if isinstance(payload, list):
        lines: List[str] = []
        for idx, item in enumerate(payload, 1):
            if isinstance(item, dict):
                snippet = item.get("search_result") or item.get("result") or item
                dataset_name = item.get("dataset_name") or item.get("dataset", "")
                lines.append(f"Result {idx}{f' ({dataset_name})' if dataset_name else ''}:")
                lines.append(json.dumps(snippet, ensure_ascii=False, indent=2))
            else:
                lines.append(f"Result {idx}: {item}")
            lines.append("")
        return "\n".join(line for line in lines).strip() or json.dumps(payload, indent=2)

    return json.dumps(payload, ensure_ascii=False, indent=2)
