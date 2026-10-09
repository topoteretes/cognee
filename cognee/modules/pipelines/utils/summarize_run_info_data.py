from typing import Any

from cognee.modules.data.models import Data

# ``run_info["data"]`` is audit-only metadata: nothing consumes its contents.
# Its one reader, ``cognee/modules/cognify/recovery.py``, copies the STARTED
# row's value verbatim onto the ERRORED row when it closes an abandoned run;
# the rollback itself derives what to undo from graph provenance and the
# ``Node``/``Edge`` tables, never from here. Persisting the full stringified
# payload makes the ``pipeline_runs`` table grow without bound, because large
# inputs (e.g. raw text passed to ``add``/``cognify``) are stored verbatim on
# every run. Keep a bounded preview instead so a single run cannot balloon the
# table.
MAX_RUN_INFO_DATA_CHARS = 512

# The same reasoning applies to lists of ``Data`` records: ``cognify`` passes
# every record of the dataset, so an unbounded id list makes each run's
# ``run_info`` grow with the corpus. 25 stringified UUIDs plus the marker
# serialize to roughly 1 KB — a fixed budget for this branch, not a match for
# the character cap above, which bounds a different shape of payload.
MAX_RUN_INFO_IDS = 25


def summarize_run_info_data(data: Any):
    """Return a compact, size-bounded description of pipeline-run input data.

    Lists of ``Data`` records are reduced to a bounded preview of their ids; any
    other payload is stringified and truncated so a single pipeline run cannot
    persist an arbitrarily large ``run_info`` blob.
    """
    if not data:
        return "None"
    if isinstance(data, list) and all(isinstance(item, Data) for item in data):
        ids = [str(item.id) for item in data]
        if len(ids) > MAX_RUN_INFO_IDS:
            return ids[:MAX_RUN_INFO_IDS] + [f"... [truncated, {len(ids)} ids total]"]
        return ids

    text = str(data)
    if len(text) > MAX_RUN_INFO_DATA_CHARS:
        return f"{text[:MAX_RUN_INFO_DATA_CHARS]}... [truncated, {len(text)} chars total]"
    return text
