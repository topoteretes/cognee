from typing import Any

from cognee.modules.data.models import Data

# ``run_info["data"]`` is audit-only metadata that is never read back from the
# database. Persisting the full stringified payload makes the ``pipeline_runs``
# table grow without bound, because large inputs (e.g. raw text passed to
# ``add``/``cognify``) are stored verbatim on every run. Keep a bounded preview
# instead so a single run cannot balloon the table.
MAX_RUN_INFO_DATA_CHARS = 512
# Same bound applied to the id-list branch: every Data id of the dataset is
# persisted on every run otherwise, so ``pipeline_runs`` grows with
# corpus_size * run_count even when each run only touches existing data.
MAX_RUN_INFO_DATA_IDS = 50


def summarize_run_info_data(data: Any):
    """Return a compact, size-bounded description of pipeline-run input data.

    Lists of ``Data`` records are reduced to a bounded sample of their ids; any
    other payload is stringified and truncated so a single pipeline run cannot
    persist an arbitrarily large ``run_info`` blob.
    """
    if not data:
        return "None"
    if isinstance(data, list) and all(isinstance(item, Data) for item in data):
        ids = [str(item.id) for item in data]
        if len(ids) > MAX_RUN_INFO_DATA_IDS:
            return ids[:MAX_RUN_INFO_DATA_IDS] + [
                f"... [{len(ids)} ids total]"
            ]
        return ids

    text = str(data)
    if len(text) > MAX_RUN_INFO_DATA_CHARS:
        return f"{text[:MAX_RUN_INFO_DATA_CHARS]}... [truncated, {len(text)} chars total]"
    return text
