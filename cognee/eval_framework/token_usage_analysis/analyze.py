"""Estimate the token cost of cognee memory vs. full-context prompting.

Measure a few representative chunks of an input, then extrapolate the
full-context-vs-cognee cost comparison to the whole corpus. See README.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from cli import parse_args
from corpus import read_source, sampled_chunks_from
from measure import run_measurements
from report import build_report

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")  # cognee reads .env; also exposes provider keys
    args = parse_args()

    text = read_source(args)
    sampled_chunks = sampled_chunks_from(text, args)
    all_measurements = run_measurements(sampled_chunks, args.llm_models)
    report = build_report(all_measurements, text, args)

    write_outputs(report, args)


def write_outputs(report: dict, args) -> None:
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    if not args.plot:
        return
    from plot import write_plots

    for path in write_plots(report, args.plot_dir, args.out.stem):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
