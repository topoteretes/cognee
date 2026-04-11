"""Disconnect from Cognee Cloud and revert to local mode."""

from cognee.shared.logging_utils import get_logger

logger = get_logger("serve.disconnect")


async def disconnect(clear_saved: bool = False) -> None:
    """Disconnect from Cognee Cloud and revert to local mode.

    After calling this, all V2 operations (remember, recall, improve,
    forget) will execute locally again.

    Args:
        clear_saved: If True, also delete the saved credentials at
            ``~/.cognee/cloud_credentials.json``. By default credentials
            are preserved so ``cognee.serve()`` can reconnect without
            re-authenticating.
    """
    from cognee.api.v1.serve.state import get_remote_client, set_remote_client

    client = get_remote_client()
    if client:
        await client.close()
        set_remote_client(None)
        logger.info("Disconnected from Cognee Cloud")
        print("  Disconnected from Cognee Cloud. Operations now run locally.")
    else:
        print("  Not connected to Cognee Cloud.")

    if clear_saved:
        from cognee.api.v1.serve.credentials import clear_credentials

        clear_credentials()
        print("  Saved credentials cleared.")
