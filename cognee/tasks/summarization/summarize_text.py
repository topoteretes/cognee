import asyncio
from typing import Optional, Type
from uuid import uuid5
from pydantic import BaseModel

from cognee.tasks.summarization.exceptions import InvalidSummaryInputsError
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.infrastructure.llm.extraction.extract_summary import extract_summary_with_provenance
from cognee.infrastructure.llm.pipeline_stage import pipeline_stage
from cognee.modules.cognify.config import get_cognify_config
from cognee.tasks.summarization.models import TextSummary


from cognee.modules.pipelines.tasks.task import task_summary

# Stage label on the summary.generated events (SDK-529).
CAPTURE_STAGE = "summarize_text"


@task_summary("Summarized {n} chunk(s)")
async def summarize_text(
    data_chunks: list[DocumentChunk], summarization_model: Optional[Type[BaseModel]] = None
):
    """
    Summarize the text contained in the provided data chunks.

    If no summarization model is provided, the function retrieves the default model from the
    configuration. It processes the data chunks asynchronously and returns summaries for
    each chunk. If the provided list of data chunks is empty, it simply returns the list as
    is.

    While eval capture is active (SDK-529) one ``summary.generated`` event per chunk is
    emitted, carrying that summary's provenance — the model id, the fingerprint of the
    summarization prompt and the fingerprint of the source chunk text — alongside the
    chunk and summary ids and a character count; never the summary or chunk text. The
    provenance is computed for the event only and is deliberately NOT stored on the
    ``TextSummary`` node: the event is keyed by ``summary_id``, so offline joins have it
    either way, and graph content must not depend on whether capture happened to be on.
    With capture off nothing is hashed and nothing is emitted.

    Parameters:
    -----------

        - data_chunks (list[DocumentChunk]): A list of DocumentChunk objects containing text
          to be summarized.
        - summarization_model (Type[BaseModel]): An optional model used for summarizing
          text. If not provided, the default is fetched from the configuration. (default
          None)

    Returns:
    --------

        A list of TextSummary objects, each containing the summary of a corresponding
        DocumentChunk.
    """

    if not isinstance(data_chunks, list):
        raise InvalidSummaryInputsError("data_chunks must be a list.")
    if not all(hasattr(c, "text") for c in data_chunks):
        raise InvalidSummaryInputsError("each DocumentChunk must have a 'text' attribute.")

    if len(data_chunks) == 0:
        return data_chunks

    if summarization_model is None:
        cognee_config = get_cognify_config()
        summarization_model = cognee_config.summarization_model

    # Lazy on purpose: ``import cognee`` must not load the capture package.
    from cognee.modules.observability import capture as eval_capture

    # One global read, hoisted out of the per-chunk loop.
    active = eval_capture.is_active()

    with pipeline_stage("summarization"):
        results = await asyncio.gather(
            *[
                extract_summary_with_provenance(chunk.text, summarization_model)
                for chunk in data_chunks
            ]
        )

    # Every chunk reads the same prompt file, so its fingerprint is computed once.
    prompt_fingerprints: dict[str, str] = {}
    summaries: list[TextSummary] = []
    # Run-wide provenance for the manifest; the stage model and prompt are the
    # same for every chunk, so the last chunk's values describe the run.
    run_model: Optional[str] = None
    run_prompt_fingerprint: Optional[str] = None

    for chunk, (llm_output, prompt_text, model_name) in zip(data_chunks, results):
        # Capture-only provenance; never stored on the node.
        prompt_fingerprint: Optional[str] = None
        source_text_hash: Optional[str] = None
        if active:
            # The sanctioned snapshot cost: sha256 of the chunk text (and of each
            # distinct prompt) — only while capturing.
            prompt_fingerprint = prompt_fingerprints.get(prompt_text)
            if prompt_fingerprint is None:
                prompt_fingerprint = eval_capture.prompt_fingerprint(prompt_text)
                prompt_fingerprints[prompt_text] = prompt_fingerprint
            source_text_hash = eval_capture.prompt_fingerprint(chunk.text)

        summary = TextSummary(
            id=uuid5(chunk.id, "TextSummary"),
            made_from=chunk,
            source_chunk_id=str(chunk.id),
            belongs_to_set=chunk.belongs_to_set,
            text=llm_output.summary,
            importance_weight=chunk.importance_weight,
        )
        summaries.append(summary)

        if active:
            _emit_summary_generated(
                summary, chunk, model_name, prompt_fingerprint, source_text_hash
            )
            run_model, run_prompt_fingerprint = model_name, prompt_fingerprint

    if active:
        # Once per run, not per chunk. Taken from the locals that produced the
        # summaries, not from the node (which no longer carries them).
        eval_capture.note("summarization.model", run_model)
        eval_capture.note("summarization.prompt_fingerprint", run_prompt_fingerprint)

    return summaries


def _emit_summary_generated(
    summary: TextSummary,
    chunk: DocumentChunk,
    model: Optional[str],
    prompt_fingerprint: Optional[str],
    source_text_hash: Optional[str],
) -> None:
    """Buffer one ``summary.generated`` event: ids, fingerprints and a size — no text."""
    from cognee.modules.observability import capture as eval_capture

    eval_capture.emit(
        eval_capture.KIND_SUMMARY_GENERATED,
        {
            "chunk_id": str(chunk.id),
            "summary_id": str(summary.id),
            "model": model,
            "prompt_fingerprint": prompt_fingerprint,
            "source_text_hash": source_text_hash,
            "summary_chars": len(summary.text),
        },
        payload_kind="json",
        stage=CAPTURE_STAGE,
    )
