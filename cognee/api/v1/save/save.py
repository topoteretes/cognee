import os
import asyncio
import json
from typing import Optional, Union, List, Dict
from uuid import UUID

from pydantic import BaseModel

from cognee.base_config import get_base_config
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.data.methods import get_authorized_existing_datasets, get_dataset_data
from cognee.infrastructure.files.utils.get_data_file_path import get_data_file_path
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.shared.logging_utils import get_logger
from cognee.api.v1.search import search
from cognee.modules.search.types import SearchType


logger = get_logger("save")


class QuestionsModel(BaseModel):
    questions: List[str]


def _sanitize_filename(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".", " ") else "_" for c in name)
    return safe.strip().replace(" ", "_")


def _dataset_dir_name(dataset) -> str:
    # Prefer readable dataset name when available, fallback to id
    if getattr(dataset, "name", None):
        return _sanitize_filename(str(dataset.name))
    return str(dataset.id)


def _file_markdown_name(data_item, used_names: set[str]) -> str:
    # Use original file name if present, else data.name
    name = getattr(data_item, "name", None) or "file"
    base = _sanitize_filename(str(name))
    filename = f"{base}.md"
    if filename in used_names:
        short_id = str(getattr(data_item, "id", ""))[:8]
        filename = f"{base}__{short_id}.md"
    used_names.add(filename)
    return filename


def _ascii_path_tree(path_str: str) -> str:
    if not path_str:
        return "(no path)"

    # Normalize special schemes but keep segments readable
    try:
        normalized = get_data_file_path(path_str)
    except Exception:
        normalized = path_str

    # Keep the path compact – show last 5 segments
    parts = [p for p in normalized.replace("\\", "/").split("/") if p]
    if len(parts) > 6:
        display = ["…"] + parts[-5:]
    else:
        display = parts

    # Render a single-branch tree
    lines = []
    for idx, seg in enumerate(display):
        prefix = "└── " if idx == 0 else ("    " * idx + "└── ")
        lines.append(f"{prefix}{seg}")
    return "\n".join(lines)


async def _get_summary_via_summaries(query_text: str, dataset_id: UUID, top_k: int) -> str:
    try:
        results = await search(
            query_text=query_text,
            query_type=SearchType.SUMMARIES,
            dataset_ids=[dataset_id],
            top_k=top_k,
        )
        if not results:
            return ""
        texts: List[str] = []
        for r in results[:top_k]:
            texts.append(str(r))
        return "\n\n".join(texts)
    except Exception as e:
        logger.error(
            "SUMMARIES search failed for '%s' in dataset %s: %s",
            query_text,
            str(dataset_id),
            str(e),
        )
        return ""


async def _generate_questions(file_name: str, summary_text: str) -> List[str]:
    prompt = (
        "You are an expert analyst. Given a file and its summary, propose 10 diverse, high-signal "
        "questions to further explore the file's content, implications, relationships, and gaps. "
        "Avoid duplicates; vary depth and angle (overview, details, cross-references, temporal, quality).\n\n"
        f"File: {file_name}\n\nSummary:\n{summary_text[:4000]}"
    )

    model = await LLMGateway.acreate_structured_output(
        text_input=prompt,
        system_prompt="Return strictly a JSON with key 'questions' and value as an array of 10 concise strings.",
        response_model=QuestionsModel,
    )

    # model can be either pydantic model or dict-like, normalize
    try:
        questions = list(getattr(model, "questions", []))
    except Exception:
        questions = []

    # Fallback if the tool returned a dict-like
    if not questions and isinstance(model, dict):
        questions = list(model.get("questions", []) or [])

    # Enforce 10 max
    return questions[:10]


async def _run_searches_for_question(
    question: str, dataset_id: UUID, search_types: List[SearchType], top_k: int
) -> Dict[str, Union[str, List[dict], List[str]]]:
    async def run_one(st: SearchType):
        try:
            result = await search(
                query_text=question,
                query_type=st,
                dataset_ids=[dataset_id],
                top_k=top_k,
            )
            return st.value, result
        except Exception as e:
            logger.error("Search failed for type %s: %s", st.value, str(e))
            return st.value, [f"Error: {str(e)}"]

    pairs = await asyncio.gather(*[run_one(st) for st in search_types])
    return {k: v for k, v in pairs}


def _format_results_md(results: Dict[str, Union[str, List[dict], List[str]]]) -> str:
    lines: List[str] = []
    for st, payload in results.items():
        lines.append(f"#### {st}")
        if isinstance(payload, list):
            # Printed as bullet items; stringify dicts
            for item in payload[:5]:
                if isinstance(item, dict):
                    # compact representation
                    snippet = json.dumps(item, ensure_ascii=False)[:800]
                    lines.append(f"- {snippet}")
                else:
                    text = str(item)
                    lines.append(f"- {text[:800]}")
        else:
            lines.append(str(payload))
        lines.append("")
    return "\n".join(lines)


async def save(
    datasets: Optional[Union[List[str], List[UUID]]] = None,
    export_root_directory: Optional[str] = None,
    user: Optional[User] = None,
    # Configurable knobs
    max_questions: int = 10,
    search_types: Optional[List[Union[str, SearchType]]] = None,
    top_k: int = 5,
    include_summary: bool = True,
    include_ascii_tree: bool = True,
    concurrency: int = 4,
    timeout: Optional[float] = None,
) -> Dict[str, str]:
    """
    Export per-dataset markdown summaries and search insights for each ingested file.

    For every dataset the user can read:
    - Create a folder under export_root_directory (or data_root_directory/exports)
    - For each data item (file), create a .md containing:
      - Summary of the file (from existing TextSummary nodes)
      - A small ASCII path tree showing its folder position
      - Up to N LLM-generated question ideas (configurable)
      - Results of configured Cognee searches per question
    Also creates an index.md per dataset with links to files and an optional dataset summary.

    Returns a mapping of dataset_id -> export_directory path.
    """
    base_config = get_base_config()
    export_root = export_root_directory or os.path.join(
        base_config.data_root_directory, "memory_export"
    )
    os.makedirs(export_root, exist_ok=True)

    if user is None:
        user = await get_default_user()

    datasets_list = await get_authorized_existing_datasets(datasets, "read", user)
    results: Dict[str, str] = {}

    for dataset in datasets_list:
        ds_dir = os.path.join(export_root, _dataset_dir_name(dataset))
        os.makedirs(ds_dir, exist_ok=True)
        results[str(dataset.id)] = ds_dir

        data_items = await get_dataset_data(dataset.id)

        # Normalize search types
        if not search_types:
            effective_search_types = [
                SearchType.GRAPH_COMPLETION,
                SearchType.INSIGHTS,
                SearchType.CHUNKS,
            ]
        else:
            effective_search_types = []
            for st in search_types:
                if isinstance(st, SearchType):
                    effective_search_types.append(st)
                else:
                    try:
                        effective_search_types.append(SearchType[str(st)])
                    except Exception:
                        logger.warning("Unknown search type '%s', skipping", str(st))

        sem = asyncio.Semaphore(max(1, int(concurrency)))
        used_names: set[str] = set()
        index_entries: List[tuple[str, str]] = []

        async def process_one(data_item):
            async with sem:
                file_label = getattr(data_item, "name", str(data_item.id))
                original_path = getattr(data_item, "original_data_location", None)

                ascii_tree = (
                    _ascii_path_tree(original_path or file_label) if include_ascii_tree else ""
                )

                summary_text = ""
                if include_summary:
                    # Use SUMMARIES search scoped to dataset to derive file summary
                    file_query = getattr(data_item, "name", str(data_item.id)) or "file"
                    summary_text = await _get_summary_via_summaries(file_query, dataset.id, top_k)
                    if not summary_text:
                        summary_text = "Summary not available."

                if max_questions == 0:
                    questions = []
                else:
                    questions = await _generate_questions(file_label, summary_text)
                    if max_questions is not None and max_questions >= 0:
                        questions = questions[:max_questions]

                async def searches_for_question(q: str):
                    return await _run_searches_for_question(
                        q, dataset.id, effective_search_types, top_k
                    )

                # Run per-question searches concurrently
                per_q_results = await asyncio.gather(*[searches_for_question(q) for q in questions])

                # Build markdown content
                md_lines = [f"# {file_label}", ""]
                if include_ascii_tree:
                    md_lines.extend(["## Location", "", "```", ascii_tree, "```", ""])
                if include_summary:
                    md_lines.extend(["## Summary", "", summary_text, ""])

                md_lines.append("## Question ideas")
                for idx, q in enumerate(questions, start=1):
                    md_lines.append(f"- {idx}. {q}")
                md_lines.append("")

                md_lines.append("## Searches")
                md_lines.append("")
                for q, per_type in zip(questions, per_q_results):
                    md_lines.append(f"### Q: {q}")
                    md_lines.append(_format_results_md(per_type))
                    md_lines.append("")

                # Write to file (collision-safe)
                md_filename = _file_markdown_name(data_item, used_names)
                export_path = os.path.join(ds_dir, md_filename)
                tmp_path = export_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(md_lines))
                os.replace(tmp_path, export_path)

                index_entries.append((file_label, md_filename))

        tasks = [asyncio.create_task(process_one(item)) for item in data_items]

        if timeout and timeout > 0:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout)
            except asyncio.TimeoutError:
                logger.error("Save timed out for dataset %s", str(dataset.id))
        else:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Build dataset index.md with TOC and optional dataset summary via SUMMARIES
        try:
            index_lines = [f"# Dataset: {_dataset_dir_name(dataset)}", "", "## Files", ""]
            for display, fname in sorted(index_entries, key=lambda x: x[0].lower()):
                index_lines.append(f"- [{display}]({fname})")

            # Dataset summary section
            try:
                summaries = await search(
                    query_text="dataset overview",
                    query_type=SearchType.SUMMARIES,
                    dataset_ids=[dataset.id],
                    top_k=top_k,
                )
            except Exception as e:
                logger.error("Dataset summary search failed: %s", str(e))
                summaries = []

            if summaries:
                index_lines.extend(["", "## Dataset summary (top summaries)", ""])
                for s in summaries[:top_k]:
                    index_lines.append(f"- {str(s)[:800]}")

            with open(os.path.join(ds_dir, "index.md"), "w", encoding="utf-8") as f:
                f.write("\n".join(index_lines))
        except Exception as e:
            logger.error("Failed to write dataset index for %s: %s", str(dataset.id), str(e))

    return results
