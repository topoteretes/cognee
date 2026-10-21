from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
)

# ``run_info["data"]`` names the documents a run completed and
# ``run_info["errored_data"]`` the ones that failed in it, as lists of
# ``data_id`` strings. It used to hold every document the pipeline was handed
# (one id per corpus document on every run, #3074 / #4363) or a stringified
# preview of raw input, which put user text in the audit table. The ids come
# from the per-item ``{"run_info": ..., "data_id": ...}`` results the item
# runners yield, so an item skipped as already completed is not listed and a
# row grows with the work the run did, not with the corpus.


def summarize_run_info_data(results: list | None) -> list[str]:
    """The ``data_id``s the run completed; ``[]`` before any item has run.

    The cognee-saas pod calls this with ``None`` for the closing row of a run
    it recovers; the name stays until the pod drops that import.
    """
    return _data_ids_with_status(results, PipelineRunCompleted)


def errored_run_info_data_ids(results: list | None) -> list[str]:
    """The ``data_id``s of the items that failed in the run."""
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
