"""Errors raised by recall coverage.

All of them subclass the ``CogneeApiError`` family, so ``client.py``'s global
handler turns each into its own status code without the routes hand-rolling a
``JSONResponse``. Every constructor argument has a default: the repo's exception
contract test instantiates every family class with no arguments.

Id-keyed lookups raise 404 on an owner-scope mismatch rather than 403 — a 403
would confirm that some other owner's row with that id exists. Expected,
caller-caused errors log at WARNING so a typo'd label does not read like a
server fault in the logs.
"""

from fastapi import status

from cognee.exceptions import (
    CogneeConfigurationError,
    CogneeSystemError,
    CogneeValidationError,
)


class UnknownAgentLabelError(CogneeValidationError):
    """An ``agent_label`` that is neither in the prefix map nor a known literal.

    404 rather than an empty report: a typo must not look like "this agent asked
    nothing", which is a legitimate answer for a real label.
    """

    def __init__(
        self,
        message: str = "Unknown agent label.",
        name: str = "UnknownAgentLabelError",
        status_code: int = status.HTTP_404_NOT_FOUND,
        log_level: str = "WARNING",
    ):
        super().__init__(message, name, status_code, log_level=log_level)


class CoverageRunNotFoundError(CogneeValidationError):
    """No run with this id in the caller's owner scope."""

    def __init__(
        self,
        message: str = "Recall coverage run not found.",
        name: str = "CoverageRunNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
        log_level: str = "WARNING",
    ):
        super().__init__(message, name, status_code, log_level=log_level)


class CoverageRunInFlightError(CogneeValidationError):
    """A run for this ``(owner, agent_label)`` is already pending or running.

    Runs replay every question through search and the judge, so overlapping runs
    would multiply LLM cost and race on the same taxonomy.
    """

    def __init__(
        self,
        message: str = "A recall coverage run is already in progress for this agent.",
        name: str = "CoverageRunInFlightError",
        status_code: int = status.HTTP_409_CONFLICT,
        log_level: str = "WARNING",
    ):
        super().__init__(message, name, status_code, log_level=log_level)


class InvalidCoverageParamsError(CogneeValidationError):
    """A request carried a parameter the run does not have, or an out-of-range value.

    422 rather than a silent drop: a run that appeared to accept ``max_question``
    (singular) and then executed under the deployment default would report numbers
    the caller believes were produced under their own thresholds.
    """

    def __init__(
        self,
        message: str = "Invalid recall coverage parameters.",
        name: str = "InvalidCoverageParamsError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
        log_level: str = "WARNING",
    ):
        super().__init__(message, name, status_code, log_level=log_level)


class CoverageTopicNotFoundError(CogneeValidationError):
    """No topic with this id in the caller's owner scope."""

    def __init__(
        self,
        message: str = "Recall coverage topic not found.",
        name: str = "CoverageTopicNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
        log_level: str = "WARNING",
    ):
        super().__init__(message, name, status_code, log_level=log_level)


class SinkTopicNotEditableError(CogneeValidationError):
    """The sink is the wire literal ``"other"``, not a row, so it cannot be deleted."""

    def __init__(
        self,
        message: str = "The 'other' topic is not a stored topic and cannot be modified.",
        name: str = "SinkTopicNotEditableError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
        log_level: str = "WARNING",
    ):
        super().__init__(message, name, status_code, log_level=log_level)


class CoverageSuggestionNotFoundError(CogneeValidationError):
    """No topic suggestion with this id in the caller's owner scope."""

    def __init__(
        self,
        message: str = "Recall coverage topic suggestion not found.",
        name: str = "CoverageSuggestionNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
        log_level: str = "WARNING",
    ):
        super().__init__(message, name, status_code, log_level=log_level)


class CoverageSuggestionNotPendingError(CogneeValidationError):
    """Accept and dismiss apply to a pending suggestion only.

    Re-accepting would mint a second topic id for the same cluster and break the
    stability that accepted ids exist to provide.
    """

    def __init__(
        self,
        message: str = "Only a pending topic suggestion can be accepted or dismissed.",
        name: str = "CoverageSuggestionNotPendingError",
        status_code: int = status.HTTP_409_CONFLICT,
        log_level: str = "WARNING",
    ):
        super().__init__(message, name, status_code, log_level=log_level)


class CuratedQuestionNotFoundError(CogneeValidationError):
    """No curated question with this id in the caller's owner scope."""

    def __init__(
        self,
        message: str = "Curated question not found.",
        name: str = "CuratedQuestionNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
        log_level: str = "WARNING",
    ):
        super().__init__(message, name, status_code, log_level=log_level)


class DuplicateCuratedQuestionError(CogneeValidationError):
    """The same question text already exists in this scope (casefold-exact)."""

    def __init__(
        self,
        message: str = "This curated question already exists in this scope.",
        name: str = "DuplicateCuratedQuestionError",
        status_code: int = status.HTTP_409_CONFLICT,
        log_level: str = "WARNING",
    ):
        super().__init__(message, name, status_code, log_level=log_level)


class EmptyCuratedQuestionError(CogneeValidationError):
    """A curated question with no question text.

    Rejected rather than stored: an empty question embeds to a meaningless
    vector, would be replicated into every dataset partition, and would drag
    ``benchmark_score_pct`` down with rows nobody can answer.
    """

    def __init__(
        self,
        message: str = "A curated question needs non-empty question text.",
        name: str = "EmptyCuratedQuestionError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
        log_level: str = "WARNING",
    ):
        super().__init__(message, name, status_code, log_level=log_level)


class InvalidCuratedQuestionScopeError(CogneeValidationError):
    """``scope="agent"`` needs an ``agent_label``; ``scope="shared"`` forbids one."""

    def __init__(
        self,
        message: str = "Curated question scope and agent label are inconsistent.",
        name: str = "InvalidCuratedQuestionScopeError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
        log_level: str = "WARNING",
    ):
        super().__init__(message, name, status_code, log_level=log_level)


class EmbeddingFingerprintMismatchError(CogneeConfigurationError):
    """A stored topic centroid was embedded by a different model or dimension.

    Fails the run instead of re-embedding silently: comparing vectors from two
    embedding spaces produces confident nonsense, and topic ids are supposed to
    carry a score trend across runs.
    """

    def __init__(
        self,
        message: str = (
            "Stored topic centroids were embedded with a different model or dimension "
            "than the live embedding engine."
        ),
        name: str = "EmbeddingFingerprintMismatchError",
        status_code: int = status.HTTP_409_CONFLICT,
    ):
        super().__init__(message, name, status_code)


class DegenerateEmbeddingError(CogneeSystemError):
    """Every embedding came back with a zero norm.

    ``MOCK_EMBEDDING=true`` yields all-zero vectors, under which dedup silently
    finds nothing and every question looks unique. Raise rather than report a
    fabricated coverage number.
    """

    def __init__(
        self,
        message: str = "All question embeddings have zero norm; embeddings are unusable.",
        name: str = "DegenerateEmbeddingError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(message, name, status_code)
