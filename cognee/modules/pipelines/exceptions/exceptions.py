from cognee.exceptions import CogneeSystemError
from fastapi import status


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

    Carries the failure taxonomy so day-0 users see what broke and how to fix
    it instead of a silently "errored" result object:

    Attributes:
        dataset_name / dataset_id: the dataset whose build failed.
        error_category: taxonomy bucket (auth / provider_quota / schema /
            config / loader / db_init / unknown).
        error_class: the underlying exception class name.
        error_message: PII-scrubbed message of the underlying error.
        remedy: one-line "what to do next".
        duration_seconds: wall-clock length of the failed run, when known.
        pipeline_run_id: the errored run, for cross-referencing pipeline_runs.
    """

    def __init__(
        self,
        dataset_name: str = None,
        dataset_id=None,
        error_category: str = "unknown",
        error_class: str = None,
        error_message: str = None,
        remedy: str = None,
        duration_seconds: float = None,
        pipeline_run_id=None,
    ):
        self.dataset_name = dataset_name
        self.dataset_id = dataset_id
        self.error_category = error_category
        self.error_class = error_class
        self.error_message = error_message
        self.remedy = remedy
        self.duration_seconds = duration_seconds
        self.pipeline_run_id = pipeline_run_id

        dataset_desc = f" for dataset '{dataset_name}'" if dataset_name else ""
        duration_desc = f" after {duration_seconds:.1f}s" if duration_seconds else ""
        cause_desc = f"{error_class}: {error_message}" if error_class else (error_message or "")
        message = (
            f"Cognify failed{dataset_desc}{duration_desc} [{error_category}] {cause_desc} "
            f"| Fix: {remedy or 'see the pipeline_runs record for details.'} "
            f"| Pass raise_on_error=False to get the errored run info instead of this exception."
        )
        super().__init__(message, "CognifyFailedError", status.HTTP_422_UNPROCESSABLE_CONTENT)
