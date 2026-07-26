from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from cognee.eval_framework.beam.preprocessing.conversation_preprocessing import ConversationTurn
from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine
from cognee.infrastructure.llm import get_llm_config
from cognee.infrastructure.llm.LLMGateway import LLMGateway

PROMPT_VERSION = "conversation-message-tagged-compression-percent-v1"
COMPRESSION_RETRY_VERSION = "beam-preprocess-compression-retry-v1"


class CompressedMessage(BaseModel):
    text: str = Field(description="The shortened message.")


@dataclass(frozen=True)
class BatchRecord:
    dataset: str
    split: str
    conversation_index: int
    conversation_id: str
    batch_index: int
    batch_number: int | None
    plan: str | None
    turns: list[ConversationTurn]


@dataclass(frozen=True)
class MessageCompressionPlan:
    role: str
    current_tokens: int
    compression_percent: int


def get_token_counter() -> tuple[Callable[[str], int], str]:
    tokenizer = getattr(get_embedding_engine(), "tokenizer", None)
    if tokenizer:
        return tokenizer.count_tokens, "embedding_tokenizer"

    return lambda text: len((text or "").split()), "word_fallback"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def serialize_chunk(chunk: dict[str, str]) -> str:
    return json.dumps(chunk, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def chunk_stats(chunk: dict[str, str], token_count: Callable[[str], int]) -> dict[str, Any]:
    serialized = serialize_chunk(chunk)
    return {
        "tokens": token_count(serialized),
        "chars": len(serialized),
        "sha256": sha256_text(serialized),
    }


def role_token_stats(chunk: dict[str, str], token_count: Callable[[str], int]) -> dict[str, int]:
    user_tokens = token_count(chunk["user"])
    assistant_tokens = token_count(chunk["assistant"])
    pair_tokens = chunk_stats(chunk, token_count)["tokens"]
    return {
        "user_tokens": user_tokens,
        "assistant_tokens": assistant_tokens,
        "pair_tokens": pair_tokens,
        "json_overhead_tokens": max(0, pair_tokens - user_tokens - assistant_tokens),
    }


def turn_to_chunk(turn: ConversationTurn) -> dict[str, str]:
    return {
        "user": turn.user.body if turn.user else "",
        "assistant": turn.assistant.body if turn.assistant else "",
    }


def rounded_percent_to_remove(tokens_to_remove: int, available_tokens: int) -> int | None:
    if available_tokens <= 0:
        return None

    raw_percent = int((tokens_to_remove * 100 + available_tokens - 1) / available_tokens)
    rounded_percent = ((raw_percent + 9) // 10) * 10
    if rounded_percent > 90:
        return None

    return max(10, rounded_percent)


def resolve_message_compression_plan(
    *,
    chunk: dict[str, str],
    token_count: Callable[[str], int],
    limit: int,
    requested_percent: int | None,
) -> list[MessageCompressionPlan]:
    stats = role_token_stats(chunk, token_count)
    over_limit_tokens = max(0, stats["pair_tokens"] - limit)
    if over_limit_tokens <= 0:
        return []

    role_tokens = {
        "user": stats["user_tokens"],
        "assistant": stats["assistant_tokens"],
    }
    ordered_roles = sorted(role_tokens.items(), key=lambda item: item[1], reverse=True)

    if requested_percent is not None:
        selected: list[MessageCompressionPlan] = []
        expected_tokens_removed = 0
        for role, current_tokens in ordered_roles:
            if current_tokens <= 0:
                continue

            selected.append(
                MessageCompressionPlan(
                    role=role,
                    current_tokens=current_tokens,
                    compression_percent=requested_percent,
                )
            )
            expected_tokens_removed += int(current_tokens * requested_percent / 100)
            if expected_tokens_removed >= over_limit_tokens:
                return selected

        return selected

    single_role_candidates: list[MessageCompressionPlan] = []
    for role, current_tokens in ordered_roles:
        compression_percent = rounded_percent_to_remove(over_limit_tokens, current_tokens)
        if compression_percent is None:
            continue

        single_role_candidates.append(
            MessageCompressionPlan(
                role=role,
                current_tokens=current_tokens,
                compression_percent=compression_percent,
            )
        )

    both_available_tokens = role_tokens["user"] + role_tokens["assistant"]
    both_percent = rounded_percent_to_remove(over_limit_tokens, both_available_tokens)

    best_single_role = min(
        single_role_candidates,
        key=lambda plan: (plan.compression_percent, -plan.current_tokens),
        default=None,
    )
    if best_single_role and both_percent and best_single_role.compression_percent <= both_percent:
        return [best_single_role]
    if best_single_role and both_percent is None:
        return [best_single_role]
    if both_percent is not None:
        return [
            MessageCompressionPlan(
                role="user",
                current_tokens=role_tokens["user"],
                compression_percent=both_percent,
            ),
            MessageCompressionPlan(
                role="assistant",
                current_tokens=role_tokens["assistant"],
                compression_percent=both_percent,
            ),
        ]

    return [
        MessageCompressionPlan(
            role=role,
            current_tokens=current_tokens,
            compression_percent=90,
        )
        for role, current_tokens in ordered_roles
        if current_tokens > 0
    ]


def output_max_tokens(
    args: argparse.Namespace, current_tokens: int, compression_percent: int
) -> int:
    expected_tokens = int(current_tokens * (100 - compression_percent) / 100)
    content_budget = expected_tokens + args.output_token_buffer
    completion_budget = (current_tokens * 3) + args.output_token_buffer
    requested = max(content_budget, completion_budget)
    configured_max = get_llm_config().llm_max_completion_tokens
    return max(1, min(requested, configured_max))


def build_compression_system_prompt(
    *,
    base_prompt: str,
    role: str,
    current_tokens: int,
    compression_percent: int,
) -> str:
    return (
        f"{base_prompt.strip()}\n\n"
        "Compression request:\n"
        f"- Message role: {role}\n"
        f"- Current message length: {current_tokens} tokens\n"
        f"- Shorten by approximately {compression_percent}%\n"
    )


def message_tag(role: str) -> str:
    return f"{role}_message"


def render_message_input(*, role: str, message: str) -> str:
    tag = message_tag(role)
    return f"<{tag}>\n{message}\n</{tag}>"


async def compress_message(
    *,
    message: str,
    plan: MessageCompressionPlan,
    args: argparse.Namespace,
    base_prompt: str,
) -> tuple[str, int]:
    system_prompt = build_compression_system_prompt(
        base_prompt=base_prompt,
        role=plan.role,
        current_tokens=plan.current_tokens,
        compression_percent=plan.compression_percent,
    )
    max_output_tokens = output_max_tokens(args, plan.current_tokens, plan.compression_percent)
    compressed = await LLMGateway.acreate_structured_output(
        text_input=render_message_input(role=plan.role, message=message),
        system_prompt=system_prompt,
        response_model=CompressedMessage,
        max_tokens=max_output_tokens,
    )
    return compressed.text, max_output_tokens


def build_turn_metadata(
    batch_record: BatchRecord,
    *,
    turn_index: int,
    turn: ConversationTurn,
) -> dict[str, Any]:
    return {
        "dataset": batch_record.dataset,
        "split": batch_record.split,
        "conversation_index": batch_record.conversation_index,
        "conversation_id": batch_record.conversation_id,
        "plan": batch_record.plan,
        "batch_index": batch_record.batch_index,
        "batch_number": batch_record.batch_number,
        "turn_index": turn_index,
        "session": turn.session,
        "turn": turn.turn,
        "time_anchor": (
            turn.user.time_anchor
            if turn.user and turn.user.time_anchor
            else turn.assistant.time_anchor
            if turn.assistant
            else None
        ),
    }


def compression_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        limit=args.limit,
        compression_percent=args.compression_percent,
        output_token_buffer=args.output_token_buffer,
    )


# compress_turn (single-conversation script) is superseded by compress_chunk_with_counts:
# preprocess.py needs llm_semaphore-bounded concurrency and Counter-based call accounting
# for its report, and this is the only one of the two original per-turn compressors that
# has that plumbing.
async def compress_chunk_with_counts(
    *,
    chunk: dict[str, str],
    token_count: Callable[[str], int],
    args: argparse.Namespace,
    base_prompt: str,
    llm_semaphore: asyncio.Semaphore,
) -> tuple[dict[str, str], dict[str, Any], Counter[str]]:
    plans = resolve_message_compression_plan(
        chunk=chunk,
        token_count=token_count,
        limit=args.limit,
        requested_percent=args.compression_percent,
    )
    compressed_chunk = dict(chunk)
    message_compressions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    compress_call_args = compression_args(args)

    for plan in plans:
        original_message = chunk[plan.role]
        counts["attempted_llm_calls"] += 1
        async with llm_semaphore:
            compressed_message, max_output_tokens = await compress_message(
                message=original_message,
                plan=plan,
                args=compress_call_args,
                base_prompt=base_prompt,
            )

        counts["successful_llm_calls"] += 1
        compressed_chunk[plan.role] = compressed_message
        message_compressions.append(
            {
                "role": plan.role,
                "input_tag": f"{plan.role}_message",
                "compression_percent": plan.compression_percent,
                "max_output_tokens": max_output_tokens,
                "before": {
                    "tokens": plan.current_tokens,
                    "chars": len(original_message),
                    "sha256": sha256_text(original_message),
                    "text": original_message,
                },
                "after": {
                    "tokens": token_count(compressed_message),
                    "chars": len(compressed_message),
                    "sha256": sha256_text(compressed_message),
                    "text": compressed_message,
                },
            }
        )

    original_role_stats = role_token_stats(chunk, token_count)
    audit = {
        "selected_roles": [plan.role for plan in plans],
        "over_limit_tokens": max(0, original_role_stats["pair_tokens"] - args.limit),
        "requested_compression_percent": args.compression_percent,
        "role_compression_percents": {plan.role: plan.compression_percent for plan in plans},
        "message_compressions": message_compressions,
    }
    return compressed_chunk, audit, counts


def after_tokens(outlier: dict[str, Any]) -> int:
    after = outlier.get("after") or {}
    return int(after.get("tokens") or 0)


def outlier_limit(outlier: dict[str, Any], default_limit: int) -> int:
    return int(outlier.get("limit") or default_limit)


def target_identity(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": item.get("dataset"),
        "split": item.get("split"),
        "conversation_index": item.get("conversation_index"),
        "plan": item.get("plan"),
        "batch_number": item.get("batch_number"),
        "turn_index": item.get("turn_index"),
    }


def previous_percent(outlier: dict[str, Any], role: str) -> int:
    compression = (outlier.get("after") or {}).get("compression") or {}
    role_percents = compression.get("role_compression_percents") or {}
    return int(role_percents.get(role) or 0)


def failed_attempt_percent(outlier: dict[str, Any], role: str) -> int | None:
    for repair in reversed(outlier.get("repair_history", [])):
        previous_after = repair.get("previous_after") or {}
        compression = previous_after.get("compression") or {}
        role_percents = compression.get("role_compression_percents") or {}
        if role in role_percents:
            return int(role_percents[role])

    return None


def starting_percent(args: argparse.Namespace, outlier: dict[str, Any], role: str) -> int:
    if args.compression_percent is not None:
        return args.compression_percent

    if args.start_percent_source == "failed":
        percent = failed_attempt_percent(outlier, role)
        if percent is not None:
            return max(args.min_compression_percent, min(args.max_compression_percent, percent))

    percent = previous_percent(outlier, role)
    if percent <= 0:
        percent = args.min_compression_percent
    return max(args.min_compression_percent, min(args.max_compression_percent, percent))


def role_percent_schedule(
    *,
    roles: list[str],
    outlier: dict[str, Any],
    args: argparse.Namespace,
) -> Iterable[dict[str, int]]:
    if args.compression_percent is not None:
        yield {role: args.compression_percent for role in roles}
        return

    start_percents = {role: starting_percent(args, outlier, role) for role in roles}
    current_percents = dict(start_percents)
    while True:
        yield dict(current_percents)
        if all(percent >= args.max_compression_percent for percent in current_percents.values()):
            return

        current_percents = {
            role: min(args.max_compression_percent, percent + args.increase_by)
            for role, percent in current_percents.items()
        }


async def compress_roles(
    *,
    source_chunk: dict[str, str],
    roles: list[str],
    outlier: dict[str, Any],
    role_percents: dict[str, int],
    token_count: Callable[[str], int],
    args: argparse.Namespace,
    base_prompt: str,
    llm_semaphore: asyncio.Semaphore,
) -> tuple[dict[str, str], dict[str, Any], int]:
    compressed_chunk = dict(source_chunk)
    message_compressions: list[dict[str, Any]] = []
    repair_args = compression_args(args)
    attempted_calls = 0

    for role in roles:
        original_message = source_chunk[role]
        compression_percent = role_percents[role]
        plan = MessageCompressionPlan(
            role=role,
            current_tokens=token_count(original_message),
            compression_percent=compression_percent,
        )
        async with llm_semaphore:
            compressed_message, max_output_tokens = await compress_message(
                message=original_message,
                plan=plan,
                args=repair_args,
                base_prompt=base_prompt,
            )
        attempted_calls += 1
        compressed_chunk[role] = compressed_message
        message_compressions.append(
            {
                "role": role,
                "input_tag": message_tag(role),
                "compression_percent": compression_percent,
                "max_output_tokens": max_output_tokens,
                "before": {
                    "tokens": plan.current_tokens,
                    "chars": len(original_message),
                    "sha256": sha256_text(original_message),
                    "text": original_message,
                },
                "after": {
                    "tokens": token_count(compressed_message),
                    "chars": len(compressed_message),
                    "sha256": sha256_text(compressed_message),
                    "text": compressed_message,
                },
            }
        )

    original_role_stats = role_token_stats(source_chunk, token_count)
    compression_audit = {
        "selected_roles": roles,
        "over_limit_tokens": max(0, original_role_stats["pair_tokens"] - args.limit),
        "requested_compression_percent": args.compression_percent,
        "role_compression_percents": {
            item["role"]: item["compression_percent"] for item in message_compressions
        },
        "message_compressions": message_compressions,
        "repair": {
            "script_version": COMPRESSION_RETRY_VERSION,
            "source": "before.chunk",
            "previous_after_tokens": after_tokens(outlier),
            "previous_status": outlier.get("status"),
        },
    }
    return compressed_chunk, compression_audit, attempted_calls


async def compress_until_within_limit(
    *,
    source_chunk: dict[str, str],
    roles: list[str],
    outlier: dict[str, Any],
    token_count: Callable[[str], int],
    args: argparse.Namespace,
    base_prompt: str,
    llm_semaphore: asyncio.Semaphore,
) -> tuple[dict[str, str], dict[str, Any], int, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    best_chunk: dict[str, str] | None = None
    best_audit: dict[str, Any] | None = None
    best_tokens: int | None = None
    attempted_calls = 0

    for role_percents in role_percent_schedule(roles=roles, outlier=outlier, args=args):
        for repeat_index in range(1, args.attempts_per_percent + 1):
            try:
                compressed_chunk, compression_audit, attempt_calls = await compress_roles(
                    source_chunk=source_chunk,
                    roles=roles,
                    outlier=outlier,
                    role_percents=role_percents,
                    token_count=token_count,
                    args=args,
                    base_prompt=base_prompt,
                    llm_semaphore=llm_semaphore,
                )
            except Exception as exc:
                attempts.append(
                    {
                        "role_compression_percents": role_percents,
                        "repeat_index": repeat_index,
                        "status": "llm_error",
                        "error": repr(exc),
                    }
                )
                continue

            attempted_calls += attempt_calls
            compressed_tokens = chunk_stats(compressed_chunk, token_count)["tokens"]
            attempt_status = (
                "compressed"
                if compressed_tokens <= outlier_limit(outlier, args.limit)
                else "over_limit"
            )
            attempts.append(
                {
                    "role_compression_percents": role_percents,
                    "repeat_index": repeat_index,
                    "status": attempt_status,
                    "tokens": compressed_tokens,
                    "attempted_llm_calls": attempt_calls,
                }
            )

            if best_tokens is None or compressed_tokens < best_tokens:
                best_chunk = compressed_chunk
                best_audit = compression_audit
                best_tokens = compressed_tokens

            if attempt_status == "compressed":
                compression_audit["repair"]["attempts"] = attempts
                return compressed_chunk, compression_audit, attempted_calls, attempts

    if best_chunk is None or best_audit is None:
        raise RuntimeError(f"All repair attempts failed for {target_identity(outlier)}")

    best_audit["repair"]["attempts"] = attempts
    return best_chunk, best_audit, attempted_calls, attempts
