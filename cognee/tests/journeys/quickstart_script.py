"""The README quickstart, as a script that reports what happened.

Executed inside a fresh virtualenv by ``test_quickstart.py`` with only the
built wheel installed. Prints one JSON object on the last line of stdout.

``--mode mock`` imports the sibling ``mock_ai.py`` (copied next to this file)
so no network or keys are needed; ``--mode llm`` uses the real providers from
the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback

FACT = (
    "Title: Quickstart note\n\nGrace Hopper wrote the first compiler, called A-0, in 1952 while "
    "working at Remington Rand."
)
QUESTION = "Who wrote the first compiler?"
EXPECTED = ("hopper",)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("mock", "llm"), default="mock")
    args = parser.parse_args()

    report: dict = {"mode": args.mode, "steps": []}
    started = time.monotonic()

    def step(name: str, **fields) -> None:
        report["steps"].append(
            {"name": name, "elapsed_s": round(time.monotonic() - started, 2), **fields}
        )

    try:
        if args.mode == "mock":
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import mock_ai  # sibling copy, no package dependency

            mock_ai.install_all()
            step("mocks_installed")

        import cognee

        step("import_cognee", version=getattr(cognee, "__version__", "unknown"))

        async def run() -> None:
            result = await cognee.remember(FACT, dataset_name="quickstart")
            step("remember", status=result.status, dataset_id=str(result.dataset_id))
            report["remember_status"] = result.status
            if result.status != "completed":
                report["remember_error"] = getattr(result, "error", None)
                return

            results = await cognee.recall(
                QUESTION, datasets=["quickstart"], session_id="quickstart-1"
            )
            texts = [getattr(r, "text", str(r)) for r in results]
            step("recall", count=len(results))
            report["recall_texts"] = texts
            report["recall_answered"] = any(
                any(token in t.lower() for token in EXPECTED) for t in texts
            )

        asyncio.run(run())
        report["ok"] = report.get("remember_status") == "completed" and bool(
            report.get("recall_answered")
        )
    except Exception as error:  # report, don't hide
        report["ok"] = False
        report["exception"] = repr(error)
        report["traceback"] = traceback.format_exc()

    report["total_elapsed_s"] = round(time.monotonic() - started, 2)
    print(json.dumps(report, default=str))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
