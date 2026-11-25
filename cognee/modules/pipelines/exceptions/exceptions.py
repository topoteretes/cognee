from uuid import UUID

from fastapi import status

from cognee.exceptions import CogneeSystemError


class PipelineRunFailedError(CogneeSystemError):
    # The first per-item exception that caused the run to fail, when known.
    # run_tasks sets it so downstream logging/classification can report the
    # ROOT cause instead of this generic wrapper's message.
    first_error: BaseException | None = None

    def __init__(
        self,
        message: str = "Pipeline run failed.",
        name: str = "PipelineRunFailedError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
    ):
        super().__init__(message, name, status_code)


class CognifyFailedError(CogneeSystemError):
    """A foreground cognify pipeline run ended ERRORED and raise_on_error is on.

    Carries the ROOT cause so day-0 users see what broke instead of a silently
    "errored" result object:

    Attributes:
        dataset_name: the dataset whose build failed.
        error_class: the underlying exception class name.
        error_message: PII-scrubbed message of the underlying error.
    """

    def __init__(
        self,
        dataset_name: str | None = None,
        error_class: str | None = None,
        error_message: str | None = None,
        hint: str | None = None,
    ):
        self.dataset_name = dataset_name
        self.error_class = error_class
        self.error_message = error_message

        dataset_desc = f" for dataset '{dataset_name}'" if dataset_name else ""
        cause_desc = f"{error_class}: {error_message}" if error_class else (error_message or "")
        # The default hint describes the errored-run-info path; run-level
        # crashes wrapped by cognify pass a hint matching what
        # raise_on_error=False actually does there (re-raise the original).
        hint = (
            hint
            or "Pass raise_on_error=False to get the errored run info instead of this exception."
        )
        message = f"Cognify failed{dataset_desc}: {cause_desc} | {hint}"
        super().__init__(message, "CognifyFailedError", status.HTTP_422_UNPROCESSABLE_CONTENT)


class AbandonedPipelineRunError(CogneeSystemError):
    """The process running a pipeline stopped before writing a terminal status.

    Written by startup recovery as the error of the ERRORED row it closes such a
    run with, after rolling the run's graph work back. ``error_class`` on that
    row is how readers tell an abandoned run from one that failed on its own.
    """

    def __init__(
        self,
        message: str = (
            "Pipeline run was abandoned: its process stopped before the run finished. "
            "Startup recovery rolled the run back; re-run the pipeline."
        ),
        name: str = "AbandonedPipelineRunError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(message, name, status_code)


class AfterRunHookError(CogneeSystemError):
    """The step after a completed pipeline run failed (cognify: the document
    structure pass or the vector store compaction).

    The runs are recorded as completed and their data is stored; the step runs
    again on the dataset's next run. ``failures`` holds ``(dataset_id,
    dataset_name, error)`` for every dataset whose step failed, ``error`` being
    the exception or the reason a returned run already carries
    (``after_run_error``), and ``__cause__`` is the first of those exceptions; ``reasons`` maps each of those
    dataset ids to its error's class and scrubbed message. ``run_info`` is what
    the blocking executor would have returned (``{dataset_id: PipelineRunInfo}``,
    each failed dataset's completed run carrying its reason as
    ``after_run_error``), so a caller that treats the stored data as success can
    still read its runs.
    """

    run_info: dict | None = None

    def __init__(self, failures: list[tuple[UUID, str, BaseException | str]]):
        from cognee.modules.operations import scrub_error_message

        self.failures = failures
        self.reasons = {
            dataset_id: error
            if isinstance(error, str)
            else f"{type(error).__name__}: {scrub_error_message(error)}"
            for dataset_id, _dataset_name, error in failures
        }
        described = "; ".join(
            f"'{dataset_name}' ({self.reasons[dataset_id]})"
            for dataset_id, dataset_name, _error in failures
        )
        super().__init__(
            "The pipeline run completed and its data is stored, but the step after it "
            f"failed for dataset {described}",
            "AfterRunHookError",
        )
