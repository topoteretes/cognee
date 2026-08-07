from pydantic import BaseModel, ConfigDict


class SessionTurnSnapshot(BaseModel):
    """Immutable session state read once and used by both lanes of a concurrent turn.

    A concurrent turn analyzes the user's message and answers it at the same time, so both
    lanes have to work from the same reading of the session. Loading it once also keeps
    the two lanes from racing each other's cache reads.
    """

    model_config = ConfigDict(frozen=True)

    raw_message: str
    # (qa_id, question, answer) for the last two turns; builds the conversational query.
    recent_qas: tuple[tuple[str, str, str], ...] = ()
    completion_history: str = ""
    active_context: str = ""
    active_context_ids: tuple[str, ...] = ()
    previous_qa_id: str | None = None
    previous_question: str | None = None
    previous_answer: str | None = None
    # (entry_id, content) served to the previous answer; the analysis lane rates these.
    previous_served_context: tuple[tuple[str, str], ...] = ()
