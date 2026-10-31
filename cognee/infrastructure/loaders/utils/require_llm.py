from cognee.infrastructure.llm.exceptions import LLMAPIKeyNotSetError


def require_llm_for_media(media: str, work: str) -> None:
    """Fail with a message about *this* file before doing work that needs the LLM.

    ``add()`` does not probe the LLM when none is configured: keyless cognee is
    a supported setup — the graph is extracted with the local GLiNER model and
    embedded with the local embedder — and a probe that can only restate the
    absent key would block that whole mode. Media files are the one input that
    still needs a key during ingestion (there is no local transcription or
    vision path), so each media loader says so itself, naming what it was about
    to do and what works without a key.

    Called before the loader reads or writes anything, so a run that cannot
    finish leaves no derived text behind.

    Args:
        media: The kind of file, as the message should name it ("Image").
        work: What the LLM is needed for, continuing "cognee " ("describes
            images with a vision model before indexing them").
    """
    from cognee.modules.preflight import llm_available

    if llm_available():
        return

    raise LLMAPIKeyNotSetError(
        f"{media} ingestion needs an LLM: {work}, and no LLM API key is configured. "
        "Set LLM_API_KEY to ingest this file. Text, document and code files need no "
        "key — cognee processes those with local models."
    )
