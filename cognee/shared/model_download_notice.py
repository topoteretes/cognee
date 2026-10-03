"""Say so when a local model is about to be downloaded on first use.

Keyless setups run on local models (the GLiNER demo extractor, the fastembed
embedder) that their libraries fetch from the Hugging Face hub the first time
they are loaded. The fetch is silent apart from a progress bar the hub library
writes to a terminal, so a first ``remember()`` that takes minutes looks hung.
Each loader asks its cache first and reports through here: a download is a
warning that names the model, its size and where it goes; a cached load stays
at info.
"""

from logging import Logger


def log_model_load(
    logger: Logger,
    *,
    model: str,
    cached: bool,
    cache_dir: str,
    size_hint: str | None,
    location_var: str,
) -> None:
    """Log a local model load: a first-use download as a warning, a cached load as info."""
    if cached:
        logger.info("Loading local model %s from %s", model, cache_dir)
        return
    logger.warning(
        "Downloading local model %s (%s) to %s. This happens once, on first use, and can take "
        "several minutes depending on the connection; later runs load it from that cache. "
        "Set %s to change the location.",
        model,
        size_hint or "size unknown",
        cache_dir,
        location_var,
    )
