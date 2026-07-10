"""Command-line interface: argument definitions and default resolution.

Kept separate so analyze.py reads as pure orchestration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from corpus import DEFAULT_MAX_CHUNK_SIZE

DESCRIPTION = "Estimate the token cost of cognee memory vs. full-context prompting."


def parse_args() -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args()
    _resolve_llm_models(args, parser)
    _require_corpus_tokens_with_text(args, parser)
    return args


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", help="text file to chunk")
    source.add_argument("--dir", help="directory of .txt files to pool and chunk")
    source.add_argument("--text", help="a single representative chunk")

    parser.add_argument("--samples", type=int, default=3, help="chunks to measure")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chunk-size", type=int, default=DEFAULT_MAX_CHUNK_SIZE)
    parser.add_argument(
        "--llm-models",
        type=_comma_list,
        default=None,
        help="comma list; default = the model cognee is configured with in .env",
    )
    parser.add_argument("--reduction-factors", type=_comma_factors, default=[1, 2, 7, 10])
    parser.add_argument(
        "--corpus-tokens",
        type=int,
        default=None,
        help="corpus size; defaults to the input's token count. Required with --text.",
    )
    parser.add_argument("--retrieved-context", type=int, default=1118)
    parser.add_argument("--query-overhead", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("token_usage_report.json"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", type=Path, default=Path("."))
    return parser


def _comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _comma_factors(value: str) -> list[float]:
    factors = [float(item) for item in _comma_list(value)]
    return [int(factor) if factor.is_integer() else factor for factor in factors]


def _resolve_llm_models(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Default to the single llm_model cognee is configured with in .env."""
    if args.llm_models:
        return
    from cognee.infrastructure.llm.config import get_llm_config

    configured = get_llm_config().llm_model
    if not configured:
        parser.error("no --llm-models given and no LLM_MODEL configured in .env")
    args.llm_models = [configured]


def _require_corpus_tokens_with_text(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if args.text is not None and args.corpus_tokens is None:
        parser.error("--corpus-tokens is required with --text (a lone chunk is not a corpus)")
