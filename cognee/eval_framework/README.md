# Cognee Evaluation Harness

Reproducible, one-command **memory-quality benchmarking** for cognee. The harness
chains four steps — **corpus building → question answering → evaluation →
dashboard** — for a single deterministic config.

The harness ships with cognee, but its heavier dependencies — the HTML dashboard,
the DeepEval engine, and some dataset downloads — live in the **optional `eval`
extra**. Core cognee never imports the harness.

## Install

```bash
pip install "cognee[eval]"
```

This one extra installs the analysis/dashboard dependencies plus the DeepEval
engine — everything needed to run a benchmark end to end. Core cognee never
imports the harness, so a plain `pip install cognee` is unaffected. The extra
itself is only needed for the HTML dashboard (plotly), the DeepEval engine
(deepeval), and downloading some benchmark datasets (e.g. Musique via gdown):
a `--engine direct_llm --no-dashboard` run works without it. When the dashboard
is enabled but its dependencies are missing, the runner fails fast before any
pipeline work with an actionable error.

## Run a benchmark in one command

CLI:

```bash
cognee eval --benchmark HotPotQA --engine direct_llm --limit 5
```

Or as a module:

```bash
python -m cognee.eval_framework --benchmark HotPotQA --engine direct_llm --limit 5
```

Key flags:

| Flag | Meaning |
| --- | --- |
| `--benchmark, -b` | Benchmark dataset (`HotPotQA`, `Musique`, `TwoWikiMultiHop`, `Dummy`, …). Once registered, `LongMemEval` works here too. |
| `--engine, -e` | `direct_llm` (uses the LLM from your `.env`) or `deepeval` (requires the `eval` extra). |
| `--limit, -n` | Number of samples in the corpus. |
| `--seed` | Seed for deterministic corpus sampling (default: `42`). |
| `--qa-engine` | Retriever used to answer questions (default: `cognee_graph_completion`). |
| `--output-dir, -o` | Directory for run artifacts, namespaced by benchmark/engine so runs stay comparable. |
| `--dashboard / --no-dashboard` | Toggle HTML dashboard generation (on by default; requires the `eval` extra). |

## Programmatic use

```python
import asyncio
from cognee.eval_framework.eval_config import EvalConfig
from cognee.eval_framework.runner import run_eval

config = EvalConfig(
    benchmark="HotPotQA",
    evaluation_engine="DirectLLM",
    number_of_samples_in_corpus=5,
    seed=42,
    results_dir="eval_results",
)

result = asyncio.run(run_eval(config))
print(result.aggregate_metrics)     # {'correctness': {'mean': ..., 'ci_lower': ..., ...}}
print(result.metrics_path)          # per-answer metrics artifact
print(result.config_path)           # resolved config, saved for reproducibility
```

`run_eval` returns an `EvalResult` with the produced artifact paths and the
aggregate metrics, so it is callable and assertable from tests and other code
instead of relying on side-effect files.

## Reproducibility

- `--seed` is threaded through the corpus builder into the benchmark adapters so
  sampling is deterministic across runs.
- When `--output-dir` is set, artifacts (questions, answers, metrics, dashboard,
  and the resolved `eval_config.json`) are written under
  `<output-dir>/<benchmark>_<engine>/`, so successive runs are comparable instead
  of overwriting each other.

## Adding a benchmark

Subclass `BaseBenchmarkAdapter` (see `benchmark_adapters/hotpot_qa_adapter.py`)
and register it in `benchmark_adapters/benchmark_adapters.py`. It then works with
the runner and CLI with no further wiring:

```bash
cognee eval --benchmark <YourBenchmark> --engine direct_llm
```
