# Attribution

These artifacts refer to questions of the [LoCoMo](https://github.com/snap-research/locomo) dataset by Maharana et al. (2024), licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), by id, and include error types from the LoCoMo audit by Penfield Labs ([dial481/locomo-audit](https://github.com/dial481/locomo-audit), CC BY-NC 4.0). They are packaged with Cognee answers and judge verdicts; see the [LoCoMo README](../README.md) for the results they support.

# Layout

- The top level holds the run of the default configuration (hybrid retrieval 20/20/20): per-question files in `qa/`, the summary tables, the ingestion reports, the gold exclusions and the run configuration.
- `topk20_run2/`: a second run of answers and judge on the same memories.
- `topk5_adapted_prompts/`: the top-k 5 configuration (resolved dates, adapted extraction and summary prompts, global context index), with its own ingestion reports and the date annotations inserted in each window, by offset into the window text built from `locomo10.json`.

Each folder has a `run_config.json` with its models, settings and prompt hashes, and a summary that `aggregate.py` regenerates from its `qa/` files.
