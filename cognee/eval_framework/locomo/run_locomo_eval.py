"""Run the LoCoMo benchmark on cognee end to end.

    uv run python -m cognee.eval_framework.locomo.run_locomo_eval [options]

Per conversation this launches two child processes with their own DATA/SYSTEM roots
(``<run-dir>/conv_NN/{data,system}``), so the ten LoCoMo conversations never see each other
and every memory stays on disk for later re-querying:

1. ``ingest`` — an overview document plus one document per 6-turn window (dated header), all
   through one ``remember()`` = add + cognify + improve — ``cognee.eval_framework.locomo.ingest``
2. ``answer`` — hybrid retrieval 20/20/20 (``configs/locomo_hybrid_20_20_20.json``), one answer
   prompt for every question (``qa_prompts/unified.txt``) and the LLM judge —
   ``cognee.eval_framework.locomo.answer``

then aggregates everything into ``<run-dir>/locomo_summary.{json,md}``, reporting the results
without the questions whose gold answer the audits in ``report_artifacts/gold_exclusions.json``
mark as corrupted.

Models: the extraction / answering model is ``--answer-model`` (env ``LLM_MODEL`` for the
children); the judge is ``--judge-model`` (env ``LOCOMO_JUDGE_MODEL``).

Smoke run (one conversation, three sessions, ~20 questions)::

    uv run python -m cognee.eval_framework.locomo.run_locomo_eval \
        --conversations 1 --max-sessions 3 --max-questions 20

Full run (all ten conversations)::

    uv run python -m cognee.eval_framework.locomo.run_locomo_eval --conversations 0-9

Re-answer an existing ingestion (ingestion is skipped when its report says completed)::

    uv run python -m cognee.eval_framework.locomo.run_locomo_eval --run-dir temp/locomo_runs/<run> \
        --conversations 0-9 --stage answer
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCOMO_DIR = REPO_ROOT / "cognee" / "eval_framework" / "locomo"
DEFAULT_RUNS_ROOT = REPO_ROOT / "temp" / "locomo_runs"
DEFAULT_DATA_PATH = REPO_ROOT / "temp" / "locomo_data" / "locomo10.json"
DEFAULT_CONFIG = LOCOMO_DIR / "configs" / "locomo_hybrid_20_20_20.json"
DEFAULT_GOLD_EXCLUSIONS = LOCOMO_DIR / "report_artifacts" / "gold_exclusions.json"

DEFAULT_ANSWER_MODEL = "openai/gpt-5-mini"
DEFAULT_JUDGE_MODEL = "openai/gpt-5.1"

STAGES = ("all", "ingest", "answer", "aggregate")


def print_step(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}", flush=True)


def parse_conversations(value: str) -> list[int]:
    indices: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))


def conversation_dir(run_dir: Path, index: int) -> Path:
    return run_dir / f"conv_{index:02d}"


def child_env(args: argparse.Namespace, conv_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "DATA_ROOT_DIRECTORY": str(conv_dir / "data"),
            "SYSTEM_ROOT_DIRECTORY": str(conv_dir / "system"),
            "ENABLE_BACKEND_ACCESS_CONTROL": "false",
            "CACHING": "true",
            "AUTO_FEEDBACK": "false",
            "LLM_MODEL": args.answer_model,
            "LOCOMO_JUDGE_MODEL": args.judge_model,
            "LOCOMO_DATA_PATH": str(args.data_path),
            "LOG_LEVEL": os.environ.get("LOG_LEVEL", "ERROR"),
            "COGNEE_LOG_FILE": "false",
            "COGNEE_CLI_MODE": "true",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.setdefault("TELEMETRY_DISABLED", "1")
    if args.llm_api_key:
        env["LLM_API_KEY"] = args.llm_api_key
    return env


def run_child(module: str, cli_args: list[str], env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", module, *cli_args]
    print_step(f"$ {' '.join(command[2:])}  (log: {log_path})")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {datetime.now(timezone.utc).isoformat()} =====\n")
        log.write(" ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            if line.startswith("["):  # the children's own print_step lines
                print("    " + line.rstrip(), flush=True)
        return process.wait()


def ingest_args(args: argparse.Namespace, index: int, conv_dir: Path) -> list[str]:
    cli = [
        "--conversation-index",
        str(index),
        "--run-dir",
        str(conv_dir),
        "--data-path",
        str(args.data_path),
        "--window-turns",
        str(args.window_turns),
        "--chunk-size",
        str(args.chunk_size),
    ]
    if args.max_sessions:
        cli += ["--max-sessions", str(args.max_sessions)]
    return cli


def answer_args(args: argparse.Namespace, index: int, conv_dir: Path) -> list[str]:
    cli = [
        "--conversation-index",
        str(index),
        "--run-dir",
        str(conv_dir),
        "--config-json-path",
        str(args.config_json_path),
        "--data-path",
        str(args.data_path),
        "--num-runs",
        str(args.num_runs),
        "--max-concurrent-questions",
        str(args.max_concurrent_questions),
    ]
    if args.max_sessions:
        cli += ["--max-sessions", str(args.max_sessions)]
    if args.max_questions:
        cli += ["--max-questions", str(args.max_questions)]
    if args.question_types:
        cli += ["--question-types", ",".join(args.question_types)]
    return cli


def ensure_dataset(args: argparse.Namespace) -> None:
    if args.data_path.exists():
        return
    from cognee.eval_framework.benchmark_adapters.locomo_adapter import LocomoAdapter

    print_step(f"Downloading LoCoMo to {args.data_path}")
    LocomoAdapter(data_path=str(args.data_path)).conversation_count()


def write_run_manifest(args: argparse.Namespace, conversations: list[int]) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "conversations": conversations,
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "config_json_path": str(args.config_json_path),
        "window_turns": args.window_turns,
        "chunk_size": args.chunk_size,
        "max_sessions": args.max_sessions,
        "max_questions": args.max_questions,
        "question_types": args.question_types,
        "num_runs": args.num_runs,
        "gold_exclusions": str(args.gold_exclusions) if args.gold_exclusions else None,
        "git_head": _git_head(),
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.run_dir / f"run_manifest_{stamp}.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_conversation(args: argparse.Namespace, index: int) -> dict[str, Any]:
    conv_dir = conversation_dir(args.run_dir, index)
    conv_dir.mkdir(parents=True, exist_ok=True)
    outcome: dict[str, Any] = {"conversation_index": index}
    started = time.monotonic()
    env = child_env(args, conv_dir)

    if args.stage in ("all", "ingest"):
        report_path = conv_dir / "ingestion_report.json"
        if report_path.exists() and not args.force_ingest:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("status") == "completed":
                print_step(
                    f"conv {index}: ingestion already completed, skipping (use --force-ingest)"
                )
                outcome["ingest"] = "skipped"
        if outcome.get("ingest") != "skipped":
            code = run_child(
                "cognee.eval_framework.locomo.ingest",
                ingest_args(args, index, conv_dir),
                env,
                conv_dir / "ingest.log",
            )
            outcome["ingest"] = "completed" if code == 0 else f"failed (exit {code})"
            if code != 0:
                outcome["seconds"] = time.monotonic() - started
                return outcome

    if args.stage in ("all", "answer"):
        code = run_child(
            "cognee.eval_framework.locomo.answer",
            answer_args(args, index, conv_dir),
            env,
            conv_dir / "answer.log",
        )
        outcome["answer"] = "completed" if code == 0 else f"failed (exit {code})"

    outcome["seconds"] = time.monotonic() - started
    return outcome


def _parse_csv(value: str) -> list[str] | None:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--conversations", default="0", help="e.g. 0, 0-2, 0,3,7 or 0-9")
    parser.add_argument(
        "--run-dir", type=Path, default=None, help="default: temp/locomo_runs/<timestamp>"
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--config-json-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--gold-exclusions",
        type=Path,
        default=DEFAULT_GOLD_EXCLUSIONS,
        help="questions with a corrupted gold answer, left out of the reported results",
    )
    parser.add_argument(
        "--answer-model", default=os.getenv("LOCOMO_ANSWER_MODEL", DEFAULT_ANSWER_MODEL)
    )
    parser.add_argument(
        "--judge-model", default=os.getenv("LOCOMO_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    )
    parser.add_argument("--llm-api-key", default=None, help="defaults to LLM_API_KEY from env/.env")
    # ingestion knobs
    parser.add_argument("--window-turns", type=int, default=6, help="turns per ingested window")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="max tokens per chunk, above the largest window, so every window stays one chunk",
    )
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument(
        "--force-ingest", action="store_true", help="re-ingest even if a report exists"
    )
    # QA knobs
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--question-types", type=_parse_csv, default=None)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--max-concurrent-questions", type=int, default=8)
    parser.add_argument("--parallel-conversations", type=int, default=1)
    args = parser.parse_args()

    if args.run_dir is None:
        args.run_dir = DEFAULT_RUNS_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.run_dir = args.run_dir.resolve()
    args.data_path = args.data_path.resolve()
    args.config_json_path = args.config_json_path.resolve()
    if not args.config_json_path.is_file():
        parser.error(f"missing sweep config: {args.config_json_path}")
    if args.gold_exclusions is not None and not args.gold_exclusions.is_file():
        parser.error(f"missing gold exclusions file: {args.gold_exclusions}")
    return args


def main() -> None:
    args = _parse_args()
    if not args.llm_api_key:
        args.llm_api_key = os.getenv("LLM_API_KEY") or _read_dotenv_key(REPO_ROOT / ".env")
    if not args.llm_api_key:
        raise SystemExit("LLM_API_KEY is not set (env or .env)")
    os.environ["LLM_API_KEY"] = args.llm_api_key

    conversations = parse_conversations(args.conversations)
    print_step(
        f"LoCoMo run {args.run_dir.name}: stage={args.stage} conversations={conversations} "
        f"answer={args.answer_model} judge={args.judge_model}"
    )
    write_run_manifest(args, conversations)

    if args.stage != "aggregate":
        ensure_dataset(args)

    outcomes: list[dict[str, Any]] = []
    if args.stage in ("all", "ingest", "answer"):
        if args.parallel_conversations > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=args.parallel_conversations) as pool:
                outcomes = list(pool.map(lambda i: run_conversation(args, i), conversations))
        else:
            for index in conversations:
                outcomes.append(run_conversation(args, index))
        for outcome in outcomes:
            print_step(f"conv {outcome['conversation_index']}: {json.dumps(outcome)}")

    if args.stage in ("all", "answer", "aggregate"):
        from cognee.eval_framework.locomo.aggregate import aggregate_and_write, render_markdown

        try:
            summary = aggregate_and_write(args.run_dir, args.gold_exclusions)
            print(render_markdown(summary))
            print_step(f"summary: {args.run_dir / 'locomo_summary.md'}")
        except FileNotFoundError as error:
            print_step(f"aggregate skipped: {error}")

    failed = [o for o in outcomes if any(str(v).startswith("failed") for v in o.values())]
    if failed:
        raise SystemExit(f"{len(failed)} conversation(s) failed; see per-conversation logs")


def _read_dotenv_key(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("LLM_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


if __name__ == "__main__":
    main()
