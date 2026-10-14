from typing import Any

from cognee.modules.data.models import Data
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunErrored,
)

# ``run_info["data"]`` is audit metadata: the three ``log_pipeline_run_*``
# writers store it and only startup recovery reads it, to copy a STARTED row's
# value onto the ERRORED row it writes. Persisting the full payload made the
# ``pipeline_runs`` table grow without bound (#3074, #4363): raw inputs were
# stored verbatim, and ``cognify`` hands every writer the dataset's whole
# ``Data`` list, most of which an incremental run skips as already completed,
# so each run stored one id per corpus document. Inputs are kept as a bounded
# preview; document lists are summarized by what the run did with them.
MAX_RUN_INFO_DATA_CHARS = 512


def summarize_run_info_data(data: Any, results: list | None = None):
    """Return a compact, size-bounded description of pipeline-run input data.

    A list of ``Data`` records becomes a dict sized by the work the run did
    (see ``summarize_run_data``); any other payload is stringified and truncated
    so a single pipeline run cannot persist an arbitrarily large blob.
    """
    if not data:
        return "None"

    if isinstance(data, list) and all(isinstance(item, Data) for item in data):
        return summarize_run_data(data, results)

    text = str(data)
    if len(text) > MAX_RUN_INFO_DATA_CHARS:
        return f"{text[:MAX_RUN_INFO_DATA_CHARS]}... [truncated, {len(text)} chars total]"
    return text


def summarize_run_data(data: list[Data], results: list | None) -> dict:
    """What a run did with a dataset's documents, sized by the work done, not the corpus.

    ``dataset_data_count`` records how many documents the dataset held. The ids
    kept are only those of the documents this run processed or failed, read from
    the per-item ``{"run_info": ..., "data_id": ...}`` results the item runners
    yield; documents the run skipped as already completed are counted, not
    listed. A steady-state run over a large dataset therefore stores a handful
    of ids rather than the whole corpus, while still naming every document it
    touched. Before any item has run (the STARTED row) only the count is known.
    """
    summary = {"dataset_data_count": len(data)}
    if results is None:
        return summary

    processed_data_ids: list[str] = []
    errored_data_ids: list[str] = []
    skipped_completed = 0
    for result in results:
        if not isinstance(result, dict):
            continue
        run_info = result.get("run_info")
        data_id = result.get("data_id")
        if isinstance(run_info, PipelineRunAlreadyCompleted):
            skipped_completed += 1
        elif isinstance(run_info, PipelineRunErrored):
            # An item that raised before naming its document has no id; its
            # error is recorded in run_info["error"] by the error writer.
            if data_id is not None:
                errored_data_ids.append(str(data_id))
        elif data_id is not None:
            processed_data_ids.append(str(data_id))

    summary["processed_data_ids"] = processed_data_ids
    summary["errored_data_ids"] = errored_data_ids
    summary["skipped_completed"] = skipped_completed
    return summary
