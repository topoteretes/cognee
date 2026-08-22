"""Step-by-step demo of the recall warm-up short-circuit (COG-6254).

What it proves, in order:

  1. COLD   — recall() against an empty graph returns an instant
              "memory_warming_up" marker: no graph search, no LLM call.
              (The demo defaults to an INVALID LLM key: if any stage tried to
              call an LLM, it would crash — that's the proof.)
  2. BURST  — a simulated heartbeat (10 recalls) against empty memory stays
              instant; this is the 511k-wasted-searches scenario removed.
  3. LEGACY — the kill switch (cognee.config.set_recall_warmup_shortcircuit)
              restores the old behavior so you can see what each cold recall
              used to cost.
  4. WARM   — remember() + recall() with a real key: the marker disappears
              immediately (cold verdicts are never cached) and real graph
              results come back.
  5. CONFIG — the runtime config surface: get()/set_recall_config()/get_all().

Run it standalone against any cognee release that includes COG-6254:

    mkdir warmup-demo && cd warmup-demo
    uv venv && source .venv/bin/activate
    uv pip install "cognee>=<release-with-COG-6254>"
    # optional, enables stage 4:
    export LLM_API_KEY="sk-...your real key..."
    uv run python recall_warmup_demo.py

Stages 1-3 and 5 need no LLM key at all. Storage is isolated in a fresh
temp directory on every run, so it never touches your existing cognee data.
"""

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

# Isolate storage BEFORE importing cognee: env vars beat .env files, so this
# wins even when run from a directory that has a cognee .env.
_ROOT = Path(tempfile.mkdtemp(prefix="cognee_warmup_demo_"))
os.environ["SYSTEM_ROOT_DIRECTORY"] = str(_ROOT / "system")
os.environ["DATA_ROOT_DIRECTORY"] = str(_ROOT / "data")

# The cold stages must work with NO working LLM. If no key is configured,
# plant an invalid one: any stage that tries an LLM call will fail loudly.
_INVALID_KEY = "sk-invalid-demo-key-no-llm-call-expected"
os.environ.setdefault("LLM_API_KEY", _INVALID_KEY)
HAS_REAL_KEY = os.environ["LLM_API_KEY"] != _INVALID_KEY

import cognee  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def describe(results) -> str:
    if not results:
        return "[] (empty result)"
    lines = []
    for entry in results:
        source = getattr(entry, "source", "?")
        status = getattr(entry, "status", None)
        text = getattr(entry, "text", None) or getattr(entry, "answer", None) or ""
        suffix = f" status={status}" if status else ""
        lines.append(f"  - source={source}{suffix}: {str(text)[:100]}")
    return "\n".join(lines)


async def timed_recall(query: str):
    started = time.perf_counter()
    try:
        results = await cognee.recall(query)
        return results, (time.perf_counter() - started) * 1000, None
    except Exception as error:
        return None, (time.perf_counter() - started) * 1000, error


async def main() -> None:
    print(f"cognee {cognee.__version__} | storage isolated in {_ROOT}")
    print(f"LLM key: {'REAL (stage 4 enabled)' if HAS_REAL_KEY else 'INVALID on purpose'}")

    # Initialize the relational DB and default user WITHOUT running any
    # pipeline — this mirrors a deployed install (server startup migrates the
    # DB) whose users haven't ingested anything yet. add()/remember() would
    # log a pipeline run and legitimately read as warm.
    from cognee.infrastructure.databases.relational import create_db_and_tables
    from cognee.modules.users.methods import get_default_user

    await create_db_and_tables()
    await get_default_user()

    banner("STAGE 1 — COLD: recall() on an empty graph returns an instant marker")
    results, elapsed_ms, error = await timed_recall("What do you know about Ada Lovelace?")
    assert error is None and results is not None, (
        f"cold recall must not raise (and must not touch the LLM): {error}"
    )
    assert len(results) == 1 and results[0].source == "system", describe(results)
    assert results[0].status == "memory_warming_up"
    print(f"took {elapsed_ms:.0f} ms (first call includes engine init), no LLM call:")
    print(describe(results))

    banner("STAGE 2 — BURST: 10 heartbeat recalls against empty memory")
    started = time.perf_counter()
    for i in range(10):
        results, _, error = await timed_recall(f"heartbeat probe {i}: anything new?")
        assert error is None and results is not None and results[0].source == "system"
    total_ms = (time.perf_counter() - started) * 1000
    print(f"10 recalls in {total_ms:.0f} ms total ({total_ms / 10:.1f} ms each).")
    print("Pre-COG-6254 each of these spun up the full search machinery — and made")
    print("an LLM call whenever the graph held any context (the production case).")

    banner("STAGE 3 — LEGACY: kill switch shows what each cold recall used to cost")
    cognee.config.set_recall_warmup_shortcircuit(False)
    results, elapsed_ms, error = await timed_recall("What do you know about Ada Lovelace?")
    if error is not None:
        print(f"took {elapsed_ms:.0f} ms, then raised {type(error).__name__}: the old path")
        print("really reached for the LLM — the exact wasted call the guard removes.")
    else:
        print(f"took {elapsed_ms:.0f} ms for the full search machinery on an EMPTY graph:")
        print(describe(results))
    cognee.config.set_recall_warmup_shortcircuit(True)

    banner("STAGE 4 — WARM: after remember(), the marker disappears immediately")
    if not HAS_REAL_KEY:
        print("SKIPPED (set a real LLM_API_KEY to run this stage).")
    else:
        await cognee.remember(
            "Ada Lovelace wrote the first computer algorithm in 1843, "
            "for Charles Babbage's Analytical Engine.",
            self_improvement=False,
        )
        # No TTL wait needed: cold verdicts are never cached, so the very
        # next recall re-probes and finds the graph warm.
        results, elapsed_ms, error = await timed_recall("Who wrote the first algorithm?")
        assert error is None and results is not None, error
        assert all(entry.source != "system" for entry in results), describe(results)
        print(f"took {elapsed_ms:.0f} ms, real graph results, no marker:")
        print(describe(results))
        results, elapsed_ms, _ = await timed_recall("Who was Ada Lovelace?")
        print(f"second warm recall took {elapsed_ms:.0f} ms (warm verdict now cached in-process).")

    banner("STAGE 5 — CONFIG: runtime surface (new in this release)")
    keys = ["recall_warmup_shortcircuit", "recall_warmup_threshold", "recall_warmup_cache_ttl"]
    print("defaults:", {key: cognee.config.get(key) for key in keys})
    cognee.config.set_recall_config({"recall_warmup_cache_ttl": "30", "recall_warmup_threshold": 2})
    print("after set_recall_config:", {key: cognee.config.get(key) for key in keys})
    assert cognee.config.get("recall_warmup_cache_ttl") == 30.0  # CLI-style strings coerce
    in_get_all = all(key in cognee.config.get_all() for key in keys)
    print(f"keys visible in cognee.config.get_all(): {in_get_all}")

    print("\nAll stages passed." if HAS_REAL_KEY else "\nCold stages passed (stage 4 skipped).")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
