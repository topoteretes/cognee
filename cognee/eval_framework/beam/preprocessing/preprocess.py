"""Preprocess BEAM/BEAM-10M conversations into audited + ingestion-ready JSON documents.

Usage example:
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run python \
        -m cognee.eval_framework.beam.preprocessing.preprocess \
        --dataset both --execute-compressions --output-dir temp/beam_preprocessed_documents
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cognee.eval_framework.beam.preprocessing.compression import (
    COMPRESSION_RETRY_VERSION,
    PROMPT_VERSION,
    BatchRecord,
    build_turn_metadata,
    chunk_stats,
    compress_chunk_with_counts,
    compress_until_within_limit,
    get_token_counter,
    outlier_limit,
    resolve_message_compression_plan,
    role_token_stats,
    turn_to_chunk,
)
from cognee.eval_framework.beam.preprocessing.conversation_preprocessing import (
    ConversationTurn,
    parse_turns_from_beam_10m_plan_batches,
    parse_turns_from_beam_batches,
)
from cognee.eval_framework.beam.preprocessing.loaders import (
    collect_beam_10m_plan_batches,
    get_beam_row,
    load_beam_10m_dataset,
    load_beam_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "beam_turn_compression_prompt.txt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "temp" / "beam_preprocessed_documents"
DEFAULT_SPLITS = ("100K", "500K", "1M")
JSON_INDENT = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_csv(value: str | None) -> list[str] | None:
    if value is None or value.strip().lower() == "all":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def maybe_limit(items: list[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return items
    return items[: max(0, limit)]


def safe_slug(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "none"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=JSON_INDENT) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_args(args: argparse.Namespace) -> None:
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.output_token_buffer < 0:
        raise ValueError("--output-token-buffer cannot be negative")
    if args.batch_concurrency < 1:
        raise ValueError("--batch-concurrency must be positive")
    if args.llm_concurrency < 1:
        raise ValueError("--llm-concurrency must be positive")
    if args.max_conversations is not None and args.max_conversations < 0:
        raise ValueError("--max-conversations cannot be negative")
    if args.max_compressions is not None and args.max_compressions < 0:
        raise ValueError("--max-compressions cannot be negative")
    if args.compression_percent is not None:
        if args.compression_percent < 10 or args.compression_percent > 90:
            raise ValueError("--compression-percent must be between 10 and 90")
        if args.compression_percent % 10 != 0:
            raise ValueError("--compression-percent must be a multiple of 10")
    if args.min_compression_percent < 10 or args.min_compression_percent > 90:
        raise ValueError("--min-compression-percent must be between 10 and 90")
    if args.max_compression_percent < args.min_compression_percent:
        raise ValueError("--max-compression-percent must be >= --min-compression-percent")
    if args.increase_by <= 0:
        raise ValueError("--increase-by must be positive")
    if args.attempts_per_percent < 1:
        raise ValueError("--attempts-per-percent must be positive")
    if args.max_chunk_size <= 0:
        raise ValueError("--max-chunk-size must be positive")


def selected_indexes(total: int, max_conversations: int | None) -> range:
    if max_conversations is None:
        return range(total)
    return range(min(total, max_conversations))


def build_beam_batch_records(
    *,
    row: dict[str, Any],
    split: str,
    conversation_index: int,
    max_batches: int | None,
) -> list[BatchRecord]:
    conversation_id = str(row.get("conversation_id", conversation_index))
    batch_records: list[BatchRecord] = []

    for batch_index, batch in enumerate(maybe_limit(row["chat"], max_batches)):
        batch_records.append(
            BatchRecord(
                dataset="BEAM",
                split=split,
                conversation_index=conversation_index,
                conversation_id=conversation_id,
                batch_index=batch_index,
                batch_number=batch_index + 1,
                plan=None,
                turns=parse_turns_from_beam_batches([batch]),
            )
        )

    return batch_records


def build_beam_10m_batch_records(
    *,
    row: dict[str, Any],
    conversation_index: int,
    plans: list[str] | None,
    max_batches_per_plan: int | None,
) -> list[BatchRecord]:
    conversation_id = str(row.get("conversation_id", conversation_index))
    plan_batches = collect_beam_10m_plan_batches(
        row["chat"],
        plans=plans,
        max_batches_per_plan=max_batches_per_plan,
    )
    batch_records: list[BatchRecord] = []

    for plan, batches in plan_batches.items():
        for batch_index, batch in enumerate(batches):
            batch_number = batch.get("batch_number", batch_index + 1)
            batch_records.append(
                BatchRecord(
                    dataset="BEAM-10M",
                    split="10M",
                    conversation_index=conversation_index,
                    conversation_id=conversation_id,
                    batch_index=batch_index,
                    batch_number=batch_number,
                    plan=plan,
                    turns=parse_turns_from_beam_10m_plan_batches([batch], plan=plan),
                )
            )

    return batch_records


def should_stop_compressing(
    args: argparse.Namespace,
    compression_state: dict[str, int],
) -> bool:
    return (
        args.max_compressions is not None
        and compression_state["compressed_turn_attempts"] >= args.max_compressions
    )


async def apply_inline_retry(
    *,
    first_attempt_chunk: dict[str, str],
    outlier: dict[str, Any],
    compression_audit: dict[str, Any],
    token_count: Callable[[str], int],
    args: argparse.Namespace,
    base_prompt: str,
    llm_semaphore: asyncio.Semaphore,
) -> tuple[dict[str, str], str, Counter[str]]:
    roles = compression_audit["selected_roles"]
    original_chunk = outlier["before"]["chunk"]
    previous_status = outlier["status"]
    previous_after = outlier["after"]

    try:
        (
            compressed_chunk,
            retry_audit,
            attempted_calls,
            _attempts,
        ) = await compress_until_within_limit(
            source_chunk=original_chunk,
            roles=roles,
            outlier=outlier,
            token_count=token_count,
            args=args,
            base_prompt=base_prompt,
            llm_semaphore=llm_semaphore,
        )
    except RuntimeError as exc:
        # Every retry attempt raised (e.g. a full LLM outage) -- keep the first attempt's
        # still-over-limit compression rather than discarding useful work for the turn.
        outlier["retry_error"] = repr(exc)
        return first_attempt_chunk, previous_status, Counter()

    compressed_stats = chunk_stats(compressed_chunk, token_count)
    final_status = (
        "compressed"
        if compressed_stats["tokens"] <= outlier_limit(outlier, args.limit)
        else "compressed_still_over_limit"
    )
    outlier.setdefault("repair_history", []).append(
        {
            "repaired_at": utc_now(),
            "script_version": COMPRESSION_RETRY_VERSION,
            "previous_status": previous_status,
            "previous_after": previous_after,
        }
    )
    outlier["status"] = final_status
    outlier["after"] = {
        **compressed_stats,
        "role_tokens": role_token_stats(compressed_chunk, token_count),
        "compression": retry_audit,
        "chunk": compressed_chunk,
    }
    outlier["error"] = None
    return (
        compressed_chunk,
        final_status,
        Counter(attempted_llm_calls=attempted_calls, successful_llm_calls=attempted_calls),
    )


async def process_turn(
    *,
    batch_record: BatchRecord,
    turn_index: int,
    turn: ConversationTurn,
    token_count: Callable[[str], int],
    args: argparse.Namespace,
    base_prompt: str,
    llm_semaphore: asyncio.Semaphore,
    compression_state: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any] | None, Counter[str]]:
    chunk = turn_to_chunk(turn)
    original_stats = chunk_stats(chunk, token_count)
    metadata = build_turn_metadata(batch_record, turn_index=turn_index, turn=turn)
    counts: Counter[str] = Counter(turn_pair_count=1)

    if original_stats["tokens"] <= args.limit:
        counts["unchanged_count"] += 1
        return (
            {
                **metadata,
                "status": "unchanged",
                "tokens": original_stats["tokens"],
                "chunk": chunk,
            },
            None,
            counts,
        )

    counts["outlier_count"] += 1
    outlier: dict[str, Any] = {
        **metadata,
        "status": "dry_run_outlier" if args.dry_run else "pending_compression",
        "prompt_version": PROMPT_VERSION,
        "limit": args.limit,
        "before": {
            **original_stats,
            "role_tokens": role_token_stats(chunk, token_count),
            "chunk": chunk,
        },
        "after": None,
        "error": None,
    }

    planned_roles = resolve_message_compression_plan(
        chunk=chunk,
        token_count=token_count,
        limit=args.limit,
        requested_percent=args.compression_percent,
    )
    counts["planned_llm_calls"] += len(planned_roles)

    if args.dry_run:
        counts["dry_run_outlier_count"] += 1
        return (
            {
                **metadata,
                "status": "dry_run_outlier",
                "tokens": original_stats["tokens"],
                "chunk": chunk,
            },
            outlier,
            counts,
        )

    if should_stop_compressing(args, compression_state):
        counts["skipped_max_compressions_count"] += 1
        outlier["status"] = "skipped_max_compressions"
        return (
            {
                **metadata,
                "status": "skipped_max_compressions",
                "tokens": original_stats["tokens"],
                "chunk": chunk,
            },
            outlier,
            counts,
        )

    compression_state["compressed_turn_attempts"] += 1
    try:
        compressed_chunk, compression_audit, compression_counts = await compress_chunk_with_counts(
            chunk=chunk,
            token_count=token_count,
            args=args,
            base_prompt=base_prompt,
            llm_semaphore=llm_semaphore,
        )
        counts.update(compression_counts)
    except Exception as exc:
        counts["compression_failed_count"] += 1
        counts["failed_compression_turns"] += 1
        outlier["status"] = "compression_failed"
        outlier["error"] = repr(exc)
        return (
            {
                **metadata,
                "status": "compression_failed",
                "tokens": original_stats["tokens"],
                "chunk": chunk,
            },
            outlier,
            counts,
        )

    compressed_stats = chunk_stats(compressed_chunk, token_count)
    final_status = (
        "compressed" if compressed_stats["tokens"] <= args.limit else "compressed_still_over_limit"
    )
    outlier["status"] = final_status
    outlier["after"] = {
        **compressed_stats,
        "role_tokens": role_token_stats(compressed_chunk, token_count),
        "compression": compression_audit,
        "chunk": compressed_chunk,
    }

    if final_status == "compressed_still_over_limit":
        compressed_chunk, final_status, retry_counts = await apply_inline_retry(
            first_attempt_chunk=compressed_chunk,
            outlier=outlier,
            compression_audit=compression_audit,
            token_count=token_count,
            args=args,
            base_prompt=base_prompt,
            llm_semaphore=llm_semaphore,
        )
        counts.update(retry_counts)

    counts[f"{final_status}_count"] += 1
    return (
        {
            **metadata,
            "status": final_status,
            "tokens": chunk_stats(compressed_chunk, token_count)["tokens"],
            "chunk": compressed_chunk,
        },
        outlier,
        counts,
    )


async def process_batch(
    *,
    batch_record: BatchRecord,
    token_count: Callable[[str], int],
    args: argparse.Namespace,
    base_prompt: str,
    batch_semaphore: asyncio.Semaphore,
    llm_semaphore: asyncio.Semaphore,
    compression_state: dict[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]], Counter[str]]:
    async with batch_semaphore:
        chunks: list[dict[str, Any]] = []
        outliers: list[dict[str, Any]] = []
        counts: Counter[str] = Counter(batch_count=1)

        for turn_index, turn in enumerate(batch_record.turns, start=1):
            if not turn.user or not turn.assistant:
                continue

            turn_payload, outlier, turn_counts = await process_turn(
                batch_record=batch_record,
                turn_index=turn_index,
                turn=turn,
                token_count=token_count,
                args=args,
                base_prompt=base_prompt,
                llm_semaphore=llm_semaphore,
                compression_state=compression_state,
            )
            chunks.append(turn_payload)
            counts.update(turn_counts)
            counts[f"status_{turn_payload['status']}"] += 1
            if outlier:
                outliers.append(outlier)

        document = {
            "document_id": (
                f"{batch_record.dataset.lower()}_conv{batch_record.conversation_index}"
                f"_plan{batch_record.plan or 'none'}_batch{batch_record.batch_number}"
            ),
            "dataset": batch_record.dataset,
            "split": batch_record.split,
            "conversation_index": batch_record.conversation_index,
            "conversation_id": batch_record.conversation_id,
            "plan": batch_record.plan,
            "batch_index": batch_record.batch_index,
            "batch_number": batch_record.batch_number,
            "chunks": chunks,
        }
        return document, outliers, counts


def summarize_counts(counts: Counter[str]) -> dict[str, Any]:
    status_counts = {
        key.removeprefix("status_"): value
        for key, value in sorted(counts.items())
        if key.startswith("status_")
    }
    return {
        "batch_count": counts["batch_count"],
        "turn_pair_count": counts["turn_pair_count"],
        "outlier_count": counts["outlier_count"],
        "dry_run_outlier_count": counts["dry_run_outlier_count"],
        "compressed_count": counts["compressed_count"],
        "compressed_still_over_limit_count": counts["compressed_still_over_limit_count"],
        "compression_failed_count": counts["compression_failed_count"],
        "skipped_max_compressions_count": counts["skipped_max_compressions_count"],
        "planned_llm_calls": counts["planned_llm_calls"],
        "attempted_llm_calls": counts["attempted_llm_calls"],
        "successful_llm_calls": counts["successful_llm_calls"],
        "failed_compression_turns": counts["failed_compression_turns"],
        "status_counts": status_counts,
        "all_outputs_within_limit": (
            counts["compressed_still_over_limit_count"]
            + counts["compression_failed_count"]
            + counts["skipped_max_compressions_count"]
            + counts["dry_run_outlier_count"]
        )
        == 0,
    }


async def preprocess_conversation(
    *,
    dataset: str,
    split: str,
    conversation_index: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    token_count: Callable[[str], int],
    token_counter: str,
    base_prompt: str,
) -> dict[str, Any]:
    plans = parse_csv(args.plans)
    if dataset == "beam":
        batch_records = build_beam_batch_records(
            row=row,
            split=split,
            conversation_index=conversation_index,
            max_batches=args.max_batches,
        )
    else:
        batch_records = build_beam_10m_batch_records(
            row=row,
            conversation_index=conversation_index,
            plans=plans,
            max_batches_per_plan=args.max_batches_per_plan,
        )

    batch_semaphore = asyncio.Semaphore(args.batch_concurrency)
    llm_semaphore = asyncio.Semaphore(args.llm_concurrency)
    compression_state = {"compressed_turn_attempts": 0}
    tasks = [
        process_batch(
            batch_record=batch_record,
            token_count=token_count,
            args=args,
            base_prompt=base_prompt,
            batch_semaphore=batch_semaphore,
            llm_semaphore=llm_semaphore,
            compression_state=compression_state,
        )
        for batch_record in batch_records
    ]
    results = await asyncio.gather(*tasks)

    documents: list[dict[str, Any]] = []
    outliers: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for document, batch_outliers, batch_counts in results:
        documents.append(document)
        outliers.extend(batch_outliers)
        counts.update(batch_counts)

    documents.sort(key=lambda item: (item["plan"] or "", item["batch_index"]))
    outliers.sort(key=lambda item: (item["plan"] or "", item["batch_index"], item["turn_index"]))

    summary = summarize_counts(counts)
    return {
        "metadata": {
            "created_at": utc_now(),
            "dataset": dataset,
            "split": split,
            "conversation_index": conversation_index,
            "conversation_id": str(row.get("conversation_id", conversation_index)),
            "plans": plans or "ALL",
            "limit": args.limit,
            "compression_percent": args.compression_percent,
            "output_token_buffer": args.output_token_buffer,
            "dry_run": args.dry_run,
            "token_counter": token_counter,
            "prompt_path": str(args.prompt_path),
            "prompt_version": PROMPT_VERSION,
            "batch_concurrency": args.batch_concurrency,
            "llm_concurrency": args.llm_concurrency,
            "max_compressions": args.max_compressions,
        },
        "summary": {
            "document_count": len(documents),
            **summary,
        },
        "documents": documents,
        "outliers": outliers,
    }


def output_path_for_conversation(
    *,
    output_dir: Path,
    dataset: str,
    split: str,
    conversation_index: int,
    plans: list[str] | None,
) -> Path:
    dataset_dir = output_dir / ("beam_10m" if dataset == "beam10m" else "beam") / split
    if dataset == "beam10m" and plans is not None:
        plan_slug = "_".join(plans)
        return dataset_dir / f"conversation_{conversation_index:06d}_{plan_slug}.json"
    return dataset_dir / f"conversation_{conversation_index:06d}.json"


def backup_path(path: Path, backup_dir: Path) -> Path:
    return backup_dir / path.relative_to(path.parents[2])


def write_conversation_json(path: Path, payload: dict[str, Any], backup_dir: Path) -> None:
    # Overwriting an existing audited conversation file (e.g. a non-resumed rerun) must not
    # silently lose the previous audit trail, so back up the original bytes first.
    if path.exists():
        target_backup_path = backup_path(path, backup_dir)
        target_backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target_backup_path)
    write_json(path, payload)


def conversation_report_entry(
    *,
    payload: dict[str, Any],
    output_path: Path,
    status: str,
) -> dict[str, Any]:
    metadata = payload["metadata"]
    summary = payload["summary"]
    return {
        "status": status,
        "output_path": str(output_path),
        "dataset": metadata["dataset"],
        "split": metadata["split"],
        "conversation_index": metadata["conversation_index"],
        "conversation_id": metadata["conversation_id"],
        "document_count": summary["document_count"],
        "batch_count": summary["batch_count"],
        "turn_pair_count": summary["turn_pair_count"],
        "outlier_count": summary["outlier_count"],
        "compressed_count": summary["compressed_count"],
        "compressed_still_over_limit_count": summary["compressed_still_over_limit_count"],
        "compression_failed_count": summary["compression_failed_count"],
        "skipped_max_compressions_count": summary["skipped_max_compressions_count"],
        "dry_run_outlier_count": summary["dry_run_outlier_count"],
        "planned_llm_calls": summary["planned_llm_calls"],
        "attempted_llm_calls": summary["attempted_llm_calls"],
        "successful_llm_calls": summary["successful_llm_calls"],
        "all_outputs_within_limit": summary["all_outputs_within_limit"],
    }


def empty_report(args: argparse.Namespace, token_counter: str) -> dict[str, Any]:
    return {
        "metadata": {
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "dataset": args.dataset,
            "splits": parse_csv(args.splits) or list(DEFAULT_SPLITS),
            "plans": parse_csv(args.plans) or "ALL",
            "output_dir": str(args.output_dir),
            "execute_compressions": args.execute_compressions,
            "dry_run": args.dry_run,
            "limit": args.limit,
            "token_counter": token_counter,
            "prompt_path": str(args.prompt_path),
            "prompt_version": PROMPT_VERSION,
            "batch_concurrency": args.batch_concurrency,
            "llm_concurrency": args.llm_concurrency,
            "max_conversations": args.max_conversations,
            "max_compressions": args.max_compressions,
            "min_compression_percent": args.min_compression_percent,
            "max_compression_percent": args.max_compression_percent,
            "increase_by": args.increase_by,
            "attempts_per_percent": args.attempts_per_percent,
            "start_percent_source": args.start_percent_source,
            "max_chunk_size": args.max_chunk_size,
            "overwrite": args.overwrite,
        },
        "summary": {},
        "conversations": [],
        "failures": [],
    }


def update_report_summary(report: dict[str, Any]) -> None:
    totals: Counter[str] = Counter()
    for item in report["conversations"]:
        totals["conversation_count"] += 1
        if item["status"] == "processed":
            totals["processed_conversation_count"] += 1
        if item["status"] == "skipped_existing":
            totals["skipped_existing_conversation_count"] += 1
        for key in (
            "document_count",
            "batch_count",
            "turn_pair_count",
            "outlier_count",
            "compressed_count",
            "compressed_still_over_limit_count",
            "compression_failed_count",
            "skipped_max_compressions_count",
            "dry_run_outlier_count",
            "planned_llm_calls",
            "attempted_llm_calls",
            "successful_llm_calls",
        ):
            totals[key] += item.get(key, 0)

    totals["failed_conversation_count"] = len(report["failures"])
    report["summary"] = {
        **dict(totals),
        "complete": totals["failed_conversation_count"] == 0,
        "all_outputs_within_limit": (
            totals["compressed_still_over_limit_count"]
            + totals["compression_failed_count"]
            + totals["skipped_max_compressions_count"]
            + totals["dry_run_outlier_count"]
        )
        == 0,
    }
    report["metadata"]["updated_at"] = utc_now()


def report_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "report.json"


def manifest_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "manifest.json"


def split_folder(split: str) -> str:
    return split.lower()


def conversation_folder(payload: dict[str, Any]) -> str:
    metadata = payload["metadata"]
    conversation_index = int(metadata["conversation_index"])
    conversation_id = safe_slug(metadata.get("conversation_id", conversation_index))
    return f"conversation_{conversation_index:06d}_id_{conversation_id}"


def session_slug(chunks: list[dict[str, Any]]) -> str:
    sessions = sorted(
        {chunk.get("session") for chunk in chunks if chunk.get("session") is not None}
    )
    if not sessions:
        return "session_unknown"
    if len(sessions) == 1:
        return f"session_{int(sessions[0]):04d}"
    return "sessions_" + "_".join(str(session) for session in sessions)


def batch_file_name(document: dict[str, Any]) -> str:
    batch_number = int(document["batch_number"])
    session = session_slug(document["chunks"])
    if document.get("plan"):
        return f"{safe_slug(document['plan'])}_batch_{batch_number:04d}_{session}.json"
    return f"batch_{batch_number:04d}_{session}.json"


def format_turn_pair(chunk: dict[str, Any]) -> str:
    turn = chunk["chunk"]
    header_parts = []
    if chunk.get("session") is not None:
        header_parts.append(f"Session: {chunk['session']}")
    if chunk.get("turn") is not None:
        header_parts.append(f"Turn: {chunk['turn']}")
    header_parts.append(f"Time anchor: {chunk.get('time_anchor') or 'unknown'}")

    header = "\n".join(header_parts)
    body = f"User:\n{turn['user']}\n\nAssistant:\n{turn['assistant']}"
    if not header:
        return body
    return f"{header}\n\n{body}"


def output_path_for_document(
    *,
    output_dir: Path,
    payload: dict[str, Any],
    document: dict[str, Any],
) -> Path:
    metadata = payload["metadata"]
    return (
        output_dir
        / split_folder(metadata["split"])
        / conversation_folder(payload)
        / batch_file_name(document)
    )


def convert_document(
    *,
    output_dir: Path,
    payload: dict[str, Any],
    document: dict[str, Any],
    token_count: Callable[[str], int],
    max_chunk_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    items = [format_turn_pair(chunk) for chunk in document["chunks"]]
    output_path = output_path_for_document(
        output_dir=output_dir, payload=payload, document=document
    )
    # Resumed conversations reach this path too (to keep manifest stats complete), so only
    # touch disk when the file is missing or the caller explicitly asked to overwrite it.
    if overwrite or not output_path.exists():
        write_json(output_path, items)

    item_sizes = [token_count(item) for item in items]
    oversized_count = sum(size > max_chunk_size for size in item_sizes)
    return {
        "path": str(output_path),
        "item_count": len(items),
        "max_item_tokens": max(item_sizes, default=0),
        "oversized_item_count": oversized_count,
        "dataset": document["dataset"],
        "split": document["split"],
        "conversation_index": document["conversation_index"],
        "conversation_id": document["conversation_id"],
        "plan": document.get("plan"),
        "batch_number": document["batch_number"],
        "sessions": sorted(
            {
                chunk.get("session")
                for chunk in document["chunks"]
                if chunk.get("session") is not None
            }
        ),
    }


def append_ingestion_ready_documents(
    *,
    output_dir: Path,
    payload: dict[str, Any],
    token_count: Callable[[str], int],
    args: argparse.Namespace,
    manifest_files: list[dict[str, Any]],
) -> None:
    for document in payload["documents"]:
        manifest_files.append(
            convert_document(
                output_dir=output_dir,
                payload=payload,
                document=document,
                token_count=token_count,
                max_chunk_size=args.max_chunk_size,
                overwrite=args.overwrite,
            )
        )


def build_manifest(
    *,
    args: argparse.Namespace,
    token_counter: str,
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    split_counts = Counter(file["split"] for file in files)
    dataset_counts = Counter(file["dataset"] for file in files)
    return {
        "metadata": {
            "created_at": utc_now(),
            "output_dir": str(args.output_dir),
            "max_chunk_size": args.max_chunk_size,
            "token_counter": token_counter,
        },
        "summary": {
            "file_count": len(files),
            "item_count": sum(file["item_count"] for file in files),
            "max_item_tokens": max((file["max_item_tokens"] for file in files), default=0),
            "oversized_item_count": sum(file["oversized_item_count"] for file in files),
            "split_counts": dict(sorted(split_counts.items())),
            "dataset_counts": dict(sorted(dataset_counts.items())),
        },
        "files": files,
    }


async def process_conversation_entry(
    *,
    dataset_name: str,
    split: str,
    conversation_index: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    token_count: Callable[[str], int],
    token_counter: str,
    base_prompt: str,
    report: dict[str, Any],
    backup_dir: Path,
    manifest_files: list[dict[str, Any]],
) -> None:
    plans = parse_csv(args.plans) if dataset_name == "beam10m" else None
    output_path = output_path_for_conversation(
        output_dir=args.output_dir,
        dataset=dataset_name,
        split=split,
        conversation_index=conversation_index,
        plans=plans,
    )

    if args.resume and output_path.exists():
        # Resumability is judged by the audited conversation file's own existence, not by
        # report.json -- report.json is derived/rebuilt data, the conversation file is the
        # source of truth for "was this conversation already processed."
        payload = read_json(output_path)
        report["conversations"].append(
            conversation_report_entry(
                payload=payload, output_path=output_path, status="skipped_existing"
            )
        )
        append_ingestion_ready_documents(
            output_dir=args.output_dir,
            payload=payload,
            token_count=token_count,
            args=args,
            manifest_files=manifest_files,
        )
        update_report_summary(report)
        write_json(report_path(args), report)
        return

    try:
        payload = await preprocess_conversation(
            dataset=dataset_name,
            split=split,
            conversation_index=conversation_index,
            row=row,
            args=args,
            token_count=token_count,
            token_counter=token_counter,
            base_prompt=base_prompt,
        )
        write_conversation_json(output_path, payload, backup_dir)
        report["conversations"].append(
            conversation_report_entry(payload=payload, output_path=output_path, status="processed")
        )
        append_ingestion_ready_documents(
            output_dir=args.output_dir,
            payload=payload,
            token_count=token_count,
            args=args,
            manifest_files=manifest_files,
        )
    except Exception as exc:
        report["failures"].append(
            {
                "dataset": dataset_name,
                "split": split,
                "conversation_index": conversation_index,
                "error": repr(exc),
            }
        )
        if args.stop_on_error:
            raise
    finally:
        update_report_summary(report)
        write_json(report_path(args), report)


async def process_beam_split(
    *,
    dataset: Any,
    split: str,
    args: argparse.Namespace,
    token_count: Callable[[str], int],
    token_counter: str,
    base_prompt: str,
    report: dict[str, Any],
    backup_dir: Path,
    manifest_files: list[dict[str, Any]],
) -> None:
    # Cross-conversation concurrency is intentionally not added here: the source scripts
    # only bound concurrency *inside* one conversation, and this merged CLI has no
    # cross-conversation concurrency flag to wire up, so conversations stay sequential.
    for conversation_index in selected_indexes(len(dataset), args.max_conversations):
        row = get_beam_row(dataset, conversation_index, dataset_label=f"BEAM {split}")
        await process_conversation_entry(
            dataset_name="beam",
            split=split,
            conversation_index=conversation_index,
            row=row,
            args=args,
            token_count=token_count,
            token_counter=token_counter,
            base_prompt=base_prompt,
            report=report,
            backup_dir=backup_dir,
            manifest_files=manifest_files,
        )


async def process_beam_10m(
    *,
    dataset: Any,
    args: argparse.Namespace,
    token_count: Callable[[str], int],
    token_counter: str,
    base_prompt: str,
    report: dict[str, Any],
    backup_dir: Path,
    manifest_files: list[dict[str, Any]],
) -> None:
    for conversation_index in selected_indexes(len(dataset), args.max_conversations):
        row = get_beam_row(dataset, conversation_index, dataset_label="BEAM-10M")
        await process_conversation_entry(
            dataset_name="beam10m",
            split="10M",
            conversation_index=conversation_index,
            row=row,
            args=args,
            token_count=token_count,
            token_counter=token_counter,
            base_prompt=base_prompt,
            report=report,
            backup_dir=backup_dir,
            manifest_files=manifest_files,
        )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    args.dry_run = not args.execute_compressions
    token_count, token_counter = get_token_counter()
    base_prompt = args.prompt_path.read_text(encoding="utf-8")
    report = empty_report(args, token_counter)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_path(args), report)
    backup_dir = args.output_dir / f"repair_backups_{timestamp()}"
    manifest_files: list[dict[str, Any]] = []

    splits = parse_csv(args.splits) or list(DEFAULT_SPLITS)
    if args.dataset in {"beam", "both"}:
        for split in splits:
            dataset = load_beam_dataset(split)
            await process_beam_split(
                dataset=dataset,
                split=split,
                args=args,
                token_count=token_count,
                token_counter=token_counter,
                base_prompt=base_prompt,
                report=report,
                backup_dir=backup_dir,
                manifest_files=manifest_files,
            )

    if args.dataset in {"beam10m", "both"}:
        dataset = load_beam_10m_dataset()
        await process_beam_10m(
            dataset=dataset,
            args=args,
            token_count=token_count,
            token_counter=token_counter,
            base_prompt=base_prompt,
            report=report,
            backup_dir=backup_dir,
            manifest_files=manifest_files,
        )

    update_report_summary(report)
    write_json(report_path(args), report)

    manifest = build_manifest(args=args, token_counter=token_counter, files=manifest_files)
    write_json(manifest_path(args), manifest)
    return {"report": report, "manifest": manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess BEAM/BEAM-10M conversations into audited + ingestion-ready JSON."
    )
    parser.add_argument("--dataset", choices=("beam", "beam10m", "both"), default="both")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--plans", default=None, help="BEAM-10M plan list, e.g. plan-1,plan-2.")
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-batches-per-plan", type=int, default=None)
    parser.add_argument("--limit", type=int, default=4096)
    parser.add_argument(
        "--compression-percent",
        type=int,
        default=None,
        help="Optional compression percent override. Must be a multiple of 10 from 10 to 90.",
    )
    parser.add_argument("--output-token-buffer", type=int, default=512)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--execute-compressions",
        action="store_true",
        help="Actually call LLMGateway. Without this flag the run is a dry-run audit.",
    )
    parser.add_argument(
        "--batch-concurrency",
        type=int,
        default=8,
        help="Maximum number of batches processed concurrently inside one conversation.",
    )
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=2,
        help="Maximum concurrent LLMGateway message-compression calls.",
    )
    parser.add_argument("--max-compressions", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--min-compression-percent", type=int, default=10)
    parser.add_argument("--max-compression-percent", type=int, default=90)
    parser.add_argument("--increase-by", type=int, default=5)
    parser.add_argument("--attempts-per-percent", type=int, default=2)
    parser.add_argument(
        "--start-percent-source",
        choices=("failed", "current"),
        default="failed",
        help="Start from the previous failed percent when available, or from the current after percent.",
    )
    parser.add_argument("--max-chunk-size", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    result = await run(args)
    report_summary = result["report"]["summary"]
    manifest_summary = result["manifest"]["summary"]
    print(f"output_dir={args.output_dir}")
    print(f"dry_run={not args.execute_compressions}")
    print(f"conversation_count={report_summary.get('conversation_count', 0)}")
    print(f"processed_conversation_count={report_summary.get('processed_conversation_count', 0)}")
    print(
        "skipped_existing_conversation_count="
        f"{report_summary.get('skipped_existing_conversation_count', 0)}"
    )
    print(f"failed_conversation_count={report_summary.get('failed_conversation_count', 0)}")
    print(f"turn_pair_count={report_summary.get('turn_pair_count', 0)}")
    print(f"outlier_count={report_summary.get('outlier_count', 0)}")
    print(f"planned_llm_calls={report_summary.get('planned_llm_calls', 0)}")
    print(f"attempted_llm_calls={report_summary.get('attempted_llm_calls', 0)}")
    print(f"successful_llm_calls={report_summary.get('successful_llm_calls', 0)}")
    print(f"compressed_count={report_summary.get('compressed_count', 0)}")
    print(
        "compressed_still_over_limit_count="
        f"{report_summary.get('compressed_still_over_limit_count', 0)}"
    )
    print(f"report={report_path(args)}")
    print(f"ingestion_ready_file_count={manifest_summary.get('file_count', 0)}")
    print(f"ingestion_ready_item_count={manifest_summary.get('item_count', 0)}")
    print(f"ingestion_ready_oversized_item_count={manifest_summary.get('oversized_item_count', 0)}")
    print(f"manifest={manifest_path(args)}")


if __name__ == "__main__":
    asyncio.run(main())
