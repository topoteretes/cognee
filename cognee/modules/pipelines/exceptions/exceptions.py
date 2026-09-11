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

    Carries the ROOT cause so day-0 users see what broke instead of a silently
    "errored" result object:

    Attributes:
        dataset_name: the dataset whose build failed.
        error_class: the underlying exception class name.
        error_message: PII-scrubbed message of the underlying error.
    """

    def __init__(
        self,
        dataset_name: str = None,
        error_class: str = None,
        error_message: str = None,
        hint: str = None,
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
