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


class AbandonedPipelineRunError(CogneeSystemError):
    """A pipeline run whose process ended before it could write a terminal status.

    A SIGKILL, an OOM kill or a pod eviction runs no Python, so the run's own
    error path never fires and its row stays DATASET_PROCESSING_STARTED.
    Startup recovery closes such a run with this class, so a consumer reading
    ``error_class`` can tell a killed run (worth re-running as-is) from one
    that genuinely failed on its input.

    Never raised and never logged by the base class: it is constructed purely
    to carry a message and an ``error_class`` onto the run's terminal row, and
    a "raised (Status code: 500)" line for an object nobody raises, with no run
    id and no dataset, is worse than no line. Recovery logs the event itself,
    with that context.
    """

    def __init__(
        self,
        pipeline_name: str | None = None,
        rolled_back: bool = False,
        name: str = "AbandonedPipelineRunError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        # Two different states share this class, and the message is what tells
        # them apart: a cognify run whose partial graph was unwound is back
        # where it started, while a run of a pipeline with no rollback policy
        # is closed with whatever it wrote still in the dataset. A reader who
        # only sees the class knows the run was killed; one who reads the
        # message knows whether anything is left behind.
        aftermath = (
            "the data it wrote was rolled back, so the dataset is back to its previous state"
            if rolled_back
            else f"{pipeline_name or 'this pipeline'} has no rollback policy, so whatever it "
            "wrote before it died is still in the dataset"
        )
        super().__init__(
            f"The {pipeline_name or 'pipeline'} run was abandoned: its process ended before "
            f"it could write a terminal status. Recovered after startup, {aftermath}. "
            "Run it again.",
            name,
            status_code,
            log=False,
        )


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
