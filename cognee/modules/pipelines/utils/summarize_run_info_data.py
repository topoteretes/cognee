from typing import Any

from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
)

# ``run_info["data"]`` names the documents a run completed, as a list of
# ``data_id`` strings — the same type the column held for ``cognify`` before.
# It used to hold every document the pipeline was handed (one id per corpus
# document on every run, #3074 / #4363) or a stringified preview of raw input,
# which put user text in the audit table. Ids come from the per-item
# ``{"run_info": ..., "data_id": ...}`` results the item runners yield, so an
# item skipped as already completed is not listed, and raw inputs (``add``)
# are named by the ``data_id`` ingestion gave them.


def summarize_run_info_data(data: Any = None, results: list | None = None) -> list[str]:
    """The ``data_id``s this run completed; empty before any item has run.

    ``data`` is deprecated and ignored: the ids are read from ``results``
    alone. It stays in the signature for callers outside core that still pass
    the run's input positionally, and will be removed once they stop.
    """
    return _data_ids_with_status(results, PipelineRunCompleted)


def errored_run_info_data_ids(results: list | None) -> list[str]:
    """The ``data_id``s of the items that failed in this run."""
    return _data_ids_with_status(results, PipelineRunErrored)


def _data_ids_with_status(results: list | None, status: type) -> list[str]:
    # An item that raised before naming its document has no id; its error is
    # recorded in run_info["error"] by the error writer.
    return [
        str(result["data_id"])
        for result in results or []
        if isinstance(result, dict)
        and isinstance(result.get("run_info"), status)
        and result.get("data_id") is not None
    ]
