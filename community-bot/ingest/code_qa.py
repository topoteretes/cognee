"""Day 2 — code knowledge ingestion via LLM Q&A pre-processing.

For each seed question we build a small context bundle from the repo
(grep for code symbols in the question, read the top matching files) and
then ask an LLM to answer grounded in that context. Each {question, answer}
pair lands in the org_community dataset; cognify folds it into the graph.

Four backends:
  --provider openai           (default) — cheap, ~$0.01 for 10 questions via
                              gpt-4o-mini + the grep-based context builder.
                              Reads LLM_API_KEY / LLM_MODEL from repo-root .env.
                              No agentic exploration — the model only sees the
                              files we pre-select by grep.
  --provider anthropic        — same grep harness, but hitting the Anthropic
                              Messages API (default claude-haiku-4-5). Reads
                              ANTHROPIC_API_KEY and optional ANTHROPIC_MODEL
                              from env. Drop-in for users on Anthropic.
  --provider anthropic-tools  — agentic: Claude with grep_repo + read_file
                              tools, drives its own investigation in a
                              tool-use loop. No CLI dependency. Better answers
                              than the grep harness, cheaper than claude-code.
  --provider claude-code      — delegate to `claude -p` subprocess. Agentic,
                              best answers, but requires the Claude Code CLI
                              on PATH and authenticated.

Why this shape: ingesting raw code produces low-signal graph nodes
("function X calls function Y"). Q&A pairs arrive already framed in
support-ticket form and grounded in specific file paths, which is what
community questions actually look like.

Usage:
    cd community-bot && python -m ingest.code_qa                     # openai
    cd community-bot && python -m ingest.code_qa --limit 3           # dry run
    cd community-bot && python -m ingest.code_qa --provider anthropic
    cd community-bot && python -m ingest.code_qa --provider anthropic-tools
    cd community-bot && python -m ingest.code_qa --provider claude-code
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cognee  # noqa: E402

from config import ORG_DATASET  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # /Users/veljko/coding/cognee

# --- Claude-code knobs -----------------------------------------------------
CLAUDE_CONCURRENCY = 3
CLAUDE_TIMEOUT_S = 240

# --- Anthropic API knobs ---------------------------------------------------
# Grep-harness branch behaves like the openai one (single call per question).
# The tool-use branch loops, so cap both concurrency and per-question time.
ANTHROPIC_CONCURRENCY = 5
ANTHROPIC_TOOLS_CONCURRENCY = 3
ANTHROPIC_TIMEOUT_S = 180
ANTHROPIC_TOOLS_MAX_TURNS = 10
ANTHROPIC_TOOL_READ_CHARS = 12000  # per read_file tool call
ANTHROPIC_TOOL_GREP_MAX_LINES = 50

# --- OpenAI knobs ----------------------------------------------------------
OPENAI_CONCURRENCY = 5  # cheaper + IO-bound, push parallelism up
OPENAI_TIMEOUT_S = 60
# Per-file excerpt cap in the context bundle (chars, not tokens — good enough)
FILE_EXCERPT_CHARS = 3500
# How many files to include in the bundle per question
MAX_FILES_IN_CONTEXT = 3
# Upper bound on matching files per symbol before we rank (per symbol, not total).
# Must be large enough that definition files in deep paths (e.g. cognee/modules/
# search/types/SearchType.py) still make it into the candidate pool when grep
# returns results sorted alphabetically. 300 is plenty; the repo has ~1k files.
MAX_GREP_MATCHES = 300

# Focused set of code-anchored questions. Edit freely.
QUESTIONS: list[str] = [
    "How does cognee.cognify() build the knowledge graph — which tasks run in what order?",
    "Where is the embedding engine selected, and how would I add a new embedding provider?",
    "Which classes implement GraphDBInterface and where do they live?",
    "How does ConversationChunker split messages into chunks? Cite the relevant file.",
    "What exact steps does add_data_points take to write nodes and edges into the graph and vector DBs?",
    "Where is the SearchType enum defined, and how does GRAPH_COMPLETION differ from GRAPH_COMPLETION_COT in the code?",
    "How does extract_graph_from_data call the LLM — what system prompt and Pydantic response model does it use?",
    "How do dataset permissions filter search results? Which function enforces the ACL?",
    "How does LiteLLMEmbeddingEngine handle batch embeddings and failures?",
    "What does index_data_points do for the vector DB, and where is it called from?",
]


# --- Keyword extraction + grep context builder -----------------------------

# Matches likely code symbols in a natural-language question.
_SYMBOL_PATTERNS = [
    re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\b"),  # dotted
    re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-zA-Z]*)+)\b"),                        # CamelCase
    re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+){1,})\b"),                      # snake_case
    re.compile(r"\b([A-Z]{2,}(?:_[A-Z0-9]+)+)\b"),                              # SCREAMING_SNAKE
]

# Paths we don't want to waste context on.
_SKIP_DIRS = ("/.venv/", "/tests/", "/notebooks/", "/dist/", "/build/", "/.git/", "/__pycache__/")


def _extract_symbols(question: str) -> list[str]:
    seen: list[str] = []
    for pat in _SYMBOL_PATTERNS:
        for m in pat.findall(question):
            if m not in seen and len(m) > 2:
                seen.append(m)
    return seen


def _grep_for_files(symbols: list[str]) -> list[Path]:
    """Return a ranked list of files most likely relevant to the symbols.

    Ranking:
      +1 point per distinct symbol that matches anywhere in the file.
      +10 bonus if the file's stem matches a symbol exactly (definition files
          win over files that only *use* the symbol — grep count otherwise
          rewards usage-heavy files like eval runners).
    """
    if not symbols:
        return []
    score: dict[Path, int] = {}
    for sym in symbols:
        try:
            proc = subprocess.run(
                ["grep", "-rln", "--include=*.py", sym, str(REPO_ROOT / "cognee")],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()][:MAX_GREP_MATCHES]
        for ln in lines:
            if any(skip in ln for skip in _SKIP_DIRS):
                continue
            p = Path(ln)
            score[p] = score.get(p, 0) + 1
            # Massive boost: this file's stem IS the symbol (definition home).
            if p.stem == sym or p.stem.lower() == sym.lower():
                score[p] = score.get(p, 0) + 10
    ranked = sorted(score.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return [p for p, _ in ranked[:MAX_FILES_IN_CONTEXT]]


def _read_excerpt(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) <= FILE_EXCERPT_CHARS:
        return text
    # Take the head — most files have imports/class defs near the top
    # which is what the model needs. Leave a marker for the truncation.
    return text[:FILE_EXCERPT_CHARS] + f"\n\n# --- truncated at {FILE_EXCERPT_CHARS} chars ---\n"


def _build_context(question: str) -> tuple[str, list[Path]]:
    symbols = _extract_symbols(question)
    files = _grep_for_files(symbols)
    if not files:
        return ("(no matching files found in cognee/ for the symbols in this question)", [])
    parts = []
    for f in files:
        rel = f.relative_to(REPO_ROOT)
        parts.append(f"### FILE: {rel}\n{_read_excerpt(f)}")
    return ("\n\n".join(parts), files)


# --- OpenAI backend --------------------------------------------------------

_OPENAI_SYSTEM = (
    "You are helping document the Cognee Python codebase for a community support "
    "knowledge base. You will be given a question and excerpts from a few repository "
    "files that were selected by lexical search. Answer the question concretely, "
    "grounded in those excerpts. Always cite file paths (e.g. cognee/foo/bar.py) "
    "and line numbers when you can see them. If the excerpts are insufficient, say "
    "so briefly instead of guessing. Plain prose, under 400 words, no markdown headers."
)


def _openai_client_and_model():
    try:
        from openai import AsyncOpenAI
    except ImportError as e:
        raise RuntimeError("openai package not installed") from e
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY / OPENAI_API_KEY not set in environment")
    # LLM_MODEL is typically "openai/gpt-4o-mini" — strip the provider prefix.
    raw_model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    model = raw_model.split("/", 1)[1] if raw_model.startswith("openai/") else raw_model
    return AsyncOpenAI(api_key=api_key), model


async def _run_openai(question: str, client, model: str, sem: asyncio.Semaphore):
    ctx, files = _build_context(question)
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"REPO EXCERPTS (selected by grep over cognee/):\n{ctx}"
    )
    async with sem:
        t0 = time.monotonic()
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _OPENAI_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.2,
                ),
                timeout=OPENAI_TIMEOUT_S,
            )
            elapsed = time.monotonic() - t0
            answer = (resp.choices[0].message.content or "").strip()
            if not answer:
                return QAResult(question, "", elapsed, False, "empty response", files)
            return QAResult(question, answer, elapsed, True, "", files)
        except asyncio.TimeoutError:
            return QAResult(question, "", time.monotonic() - t0, False, "timeout", files)
        except Exception as e:  # noqa: BLE001
            return QAResult(question, "", time.monotonic() - t0, False, repr(e)[:300], files)


# --- Anthropic backends ----------------------------------------------------

_ANTHROPIC_SYSTEM = (
    "You are helping document the Cognee Python codebase for a community support "
    "knowledge base. You will be given a question and excerpts from a few repository "
    "files that were selected by lexical search. Answer the question concretely, "
    "grounded in those excerpts. Always cite file paths (e.g. cognee/foo/bar.py) "
    "and line numbers when you can see them. If the excerpts are insufficient, say "
    "so briefly instead of guessing. Plain prose, under 400 words, no markdown headers."
)

_ANTHROPIC_TOOLS_SYSTEM = (
    "You are helping document the Cognee Python codebase for a community support "
    "knowledge base. Answer the user's question by investigating the repository: "
    "use grep_repo to locate symbols and read_file to inspect the code. Give a "
    "concrete answer citing file paths (e.g. cognee/foo/bar.py) and line numbers "
    "where helpful. If the code doesn't support a confident answer, say so briefly "
    "instead of guessing. Keep the final answer under 400 words, plain prose, no "
    "markdown headers."
)

_ANTHROPIC_TOOLS = [
    {
        "name": "grep_repo",
        "description": (
            "Run grep -rln --include=*.py over cognee/ for a pattern. Returns "
            "matching file paths (repo-relative), one per line. Use this to "
            "locate symbols, function definitions, or strings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The pattern to search for (plain string, not regex).",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a file relative to the repo root "
            "(e.g. cognee/modules/search/types/SearchType.py). Returns the first "
            f"~{ANTHROPIC_TOOL_READ_CHARS} chars of the file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to the Cognee repo root.",
                },
            },
            "required": ["path"],
        },
    },
]


def _anthropic_client_and_model():
    try:
        from anthropic import AsyncAnthropic
    except ImportError as e:
        raise RuntimeError("anthropic package not installed (pip install anthropic)") from e
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
    return AsyncAnthropic(api_key=api_key), model


def _anthropic_text(resp) -> str:
    """Pull the concatenated text from an Anthropic Messages response."""
    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


async def _run_anthropic(question: str, client, model: str, sem: asyncio.Semaphore):
    ctx, files = _build_context(question)
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"REPO EXCERPTS (selected by grep over cognee/):\n{ctx}"
    )
    async with sem:
        t0 = time.monotonic()
        try:
            resp = await asyncio.wait_for(
                client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=_ANTHROPIC_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                    temperature=0.2,
                ),
                timeout=ANTHROPIC_TIMEOUT_S,
            )
            elapsed = time.monotonic() - t0
            answer = _anthropic_text(resp)
            if not answer:
                return QAResult(question, "", elapsed, False, "empty response", files)
            return QAResult(question, answer, elapsed, True, "", files)
        except asyncio.TimeoutError:
            return QAResult(question, "", time.monotonic() - t0, False, "timeout", files)
        except Exception as e:  # noqa: BLE001
            return QAResult(question, "", time.monotonic() - t0, False, repr(e)[:300], files)


def _tool_grep_repo(pattern: str) -> str:
    if not pattern:
        return "(empty pattern)"
    try:
        proc = subprocess.run(
            ["grep", "-rln", "--include=*.py", pattern, str(REPO_ROOT / "cognee")],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return f"(grep error: {e!r})"
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    rel: list[str] = []
    repo_root_abs = REPO_ROOT.resolve()
    for ln in lines[:ANTHROPIC_TOOL_GREP_MAX_LINES]:
        try:
            rel.append(str(Path(ln).resolve().relative_to(repo_root_abs)))
        except (ValueError, OSError):
            rel.append(ln)
    if not rel:
        return "(no matches)"
    out = "\n".join(rel)
    if len(lines) > ANTHROPIC_TOOL_GREP_MAX_LINES:
        out += f"\n(+{len(lines) - ANTHROPIC_TOOL_GREP_MAX_LINES} more matches, refine pattern)"
    return out


def _tool_read_file(path: str) -> tuple[str, Path | None]:
    """Return (content_or_error, resolved_path_if_readable)."""
    if not path:
        return "(empty path)", None
    # Safety: require the resolved path to sit under REPO_ROOT.
    try:
        p = (REPO_ROOT / path).resolve()
        p.relative_to(REPO_ROOT.resolve())
    except (ValueError, OSError) as e:
        return f"(invalid path: {e!r})", None
    if not p.exists():
        return f"(file not found: {path})", None
    if not p.is_file():
        return f"(not a file: {path})", None
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"(read error: {e!r})", None
    if len(text) <= ANTHROPIC_TOOL_READ_CHARS:
        return text, p
    return (
        text[:ANTHROPIC_TOOL_READ_CHARS]
        + f"\n\n# --- truncated at {ANTHROPIC_TOOL_READ_CHARS} chars ---\n",
        p,
    )


async def _run_anthropic_tools(question: str, client, model: str, sem: asyncio.Semaphore):
    messages: list[dict] = [{"role": "user", "content": f"QUESTION: {question}"}]
    files_seen: list[Path] = []
    async with sem:
        t0 = time.monotonic()
        try:
            for _turn in range(ANTHROPIC_TOOLS_MAX_TURNS):
                resp = await asyncio.wait_for(
                    client.messages.create(
                        model=model,
                        max_tokens=2048,
                        system=_ANTHROPIC_TOOLS_SYSTEM,
                        tools=_ANTHROPIC_TOOLS,
                        messages=messages,
                        temperature=0.2,
                    ),
                    timeout=ANTHROPIC_TIMEOUT_S,
                )
                # Echo the assistant turn back in the transcript.
                messages.append({"role": "assistant", "content": resp.content})

                if resp.stop_reason != "tool_use":
                    elapsed = time.monotonic() - t0
                    answer = _anthropic_text(resp)
                    if not answer:
                        return QAResult(
                            question, "", elapsed, False, "empty response", files_seen
                        )
                    return QAResult(question, answer, elapsed, True, "", files_seen)

                tool_results = []
                for block in resp.content:
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    if block.name == "grep_repo":
                        result = _tool_grep_repo(block.input.get("pattern", ""))
                    elif block.name == "read_file":
                        result, path = _tool_read_file(block.input.get("path", ""))
                        if path is not None and path not in files_seen:
                            files_seen.append(path)
                    else:
                        result = f"(unknown tool: {block.name})"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )
                messages.append({"role": "user", "content": tool_results})

            # Ran out of turns without a final answer.
            return QAResult(
                question,
                "",
                time.monotonic() - t0,
                False,
                f"max turns ({ANTHROPIC_TOOLS_MAX_TURNS}) reached",
                files_seen,
            )
        except asyncio.TimeoutError:
            return QAResult(question, "", time.monotonic() - t0, False, "timeout", files_seen)
        except Exception as e:  # noqa: BLE001
            return QAResult(
                question, "", time.monotonic() - t0, False, repr(e)[:300], files_seen
            )


# --- Claude Code backend (kept as opt-in) ---------------------------------

_CLAUDE_PROMPT = """You are helping document the Cognee codebase for a community support knowledge base.

Your working directory is the Cognee repository root. Investigate the code directly \
(read files, grep for symbols) and answer this question concretely:

QUESTION: {question}

Requirements:
- Reference specific file paths relative to the repo root (e.g. cognee/api/v1/cognify/cognify.py).
- Include line numbers where they help (e.g. cognee/module.py:42).
- Keep the answer under 500 words of prose.
- Do NOT speculate or invent APIs — only state what you can verify in the code.
- Respond with plain text. No markdown headers. No preamble like "Here is the answer:".
"""


async def _run_claude(question: str, sem: asyncio.Semaphore):
    t0 = time.monotonic()
    async with sem:
        try:
            proc = await asyncio.create_subprocess_exec(
                "claude",
                "-p",
                _CLAUDE_PROMPT.format(question=question),
                cwd=str(REPO_ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=CLAUDE_TIMEOUT_S)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return QAResult(question, "", time.monotonic() - t0, False, "timeout", [])
            elapsed = time.monotonic() - t0
            if proc.returncode != 0:
                return QAResult(
                    question, "", elapsed, False,
                    f"exit {proc.returncode}: {(err or b'').decode(errors='replace')[:300]}",
                    [],
                )
            ans = (out or b"").decode("utf-8", errors="replace").strip()
            if not ans:
                return QAResult(question, "", elapsed, False, "empty stdout", [])
            return QAResult(question, ans, elapsed, True, "", [])
        except FileNotFoundError:
            return QAResult(question, "", time.monotonic() - t0, False, "claude not found", [])


# --- Shared pipeline -------------------------------------------------------


class QAResult(NamedTuple):
    question: str
    answer: str
    elapsed_s: float
    ok: bool
    error: str = ""
    files: list[Path] = []


def _format_qa(r: QAResult) -> str:
    refs = ""
    if r.files:
        refs = "\nFiles the model was shown: " + ", ".join(
            str(p.relative_to(REPO_ROOT)) for p in r.files
        )
    return (
        f"# Code Q&A (cognee repo)\n"
        f"Question: {r.question}\n"
        f"{refs}\n\n"
        f"Answer:\n{r.answer}\n"
    )


async def ingest_code_qa(provider: str, limit: int | None) -> int:
    questions = QUESTIONS if limit is None else QUESTIONS[:limit]

    if provider == "claude-code":
        if not shutil.which("claude"):
            print("[code_qa] ! `claude` CLI not found on PATH.")
            return 0
        sem = asyncio.Semaphore(CLAUDE_CONCURRENCY)
        print(
            f"[code_qa] provider=claude-code  questions={len(questions)}  "
            f"concurrency={CLAUDE_CONCURRENCY}  timeout={CLAUDE_TIMEOUT_S}s"
        )
        tasks = [_run_claude(q, sem) for q in questions]
    elif provider == "openai":
        client, model = _openai_client_and_model()
        sem = asyncio.Semaphore(OPENAI_CONCURRENCY)
        print(
            f"[code_qa] provider=openai  model={model}  questions={len(questions)}  "
            f"concurrency={OPENAI_CONCURRENCY}  timeout={OPENAI_TIMEOUT_S}s"
        )
        tasks = [_run_openai(q, client, model, sem) for q in questions]
    elif provider == "anthropic":
        client, model = _anthropic_client_and_model()
        sem = asyncio.Semaphore(ANTHROPIC_CONCURRENCY)
        print(
            f"[code_qa] provider=anthropic  model={model}  questions={len(questions)}  "
            f"concurrency={ANTHROPIC_CONCURRENCY}  timeout={ANTHROPIC_TIMEOUT_S}s"
        )
        tasks = [_run_anthropic(q, client, model, sem) for q in questions]
    elif provider == "anthropic-tools":
        client, model = _anthropic_client_and_model()
        sem = asyncio.Semaphore(ANTHROPIC_TOOLS_CONCURRENCY)
        print(
            f"[code_qa] provider=anthropic-tools  model={model}  questions={len(questions)}  "
            f"concurrency={ANTHROPIC_TOOLS_CONCURRENCY}  timeout={ANTHROPIC_TIMEOUT_S}s  "
            f"max_turns={ANTHROPIC_TOOLS_MAX_TURNS}"
        )
        tasks = [_run_anthropic_tools(q, client, model, sem) for q in questions]
    else:
        raise ValueError(f"unknown provider: {provider}")

    results: list[QAResult] = []
    for coro in asyncio.as_completed(tasks):
        r = await coro
        results.append(r)
        status = "OK " if r.ok else "FAIL"
        tag = f"({r.elapsed_s:5.1f}s"
        if r.files:
            tag += f", {len(r.files)} files"
        tag += ")"
        print(f"[code_qa] {status} {tag} {r.question[:70]}")
        if not r.ok:
            print(f"[code_qa]      error: {r.error}")

    ok_results = [r for r in results if r.ok]
    print(f"\n[code_qa] {len(ok_results)}/{len(results)} Q&A pairs produced.")
    if not ok_results:
        return 0

    print(f"[code_qa] Adding to dataset '{ORG_DATASET}' ...")
    for r in ok_results:
        await cognee.add(_format_qa(r), dataset_name=ORG_DATASET)

    print(f"[code_qa] Running cognify on '{ORG_DATASET}' ...")
    await cognee.cognify(datasets=[ORG_DATASET])

    print(f"[code_qa] Done. Ingested {len(ok_results)} Q&A pairs.")
    return len(ok_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest code Q&A into Cognee.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions.")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "anthropic-tools", "claude-code"],
        default="openai",
        help=(
            "LLM backend: openai (default, cheap, grep+gpt-4o-mini), anthropic "
            "(grep+Claude Messages API), anthropic-tools (agentic: Claude with "
            "grep_repo+read_file tools), or claude-code (agentic via `claude -p` "
            "CLI subprocess)."
        ),
    )
    args = parser.parse_args()
    asyncio.run(ingest_code_qa(provider=args.provider, limit=args.limit))


if __name__ == "__main__":
    main()
