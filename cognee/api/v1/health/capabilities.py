def get_capabilities() -> list[str]:
    """The optional features this server supports, as ``/health`` lists them.

    Clients read the list to choose a code path (the agent plugins cache it). A host
    that serves its own ``/health`` returns this list too, so the features each cognee
    release adds reach its clients with the release.
    """
    from cognee.infrastructure.session.capture import CAPABILITY, supported
    from cognee.modules.recall.methods.recall_index import RECALL_INDEX_CAPABILITY

    capabilities = ["session_summary_v1", "context_usage_v1", RECALL_INDEX_CAPABILITY]
    if supported():
        capabilities.append(CAPABILITY)
    return capabilities
