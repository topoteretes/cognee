"""Event hooks example: subscribe to cognee ingest/query events.

This demonstrates how an external observer (audit trail, metrics, custom
logger) can react to cognee lifecycle events without being a cognee core
dependency. Listeners can be plain functions or coroutines.

Prerequisites:
    1. Copy `.env.template` to `.env`.
    2. Set LLM_API_KEY in `.env`.
"""

import asyncio

import cognee
from cognee.api.v1.search import SearchType


def log_event(event: cognee.CogneeEvent) -> None:
    """Sync listener: print each event we observe."""
    print(f"[{event.timestamp.isoformat()}] {event.event_type} {event.payload}")


async def audit_listener(event: cognee.CogneeEvent) -> None:
    """Async listener: demonstrate that coroutine listeners are awaited."""
    if event.event_type == cognee.QUERY_AFTER:
        print(f"  -> query returned {event.payload.get('result_count', 0)} result(s)")


async def main() -> None:
    unsub_sync = cognee.subscribe(log_event)
    unsub_async = cognee.subscribe(audit_listener)

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        await cognee.add("Natural language processing is a field of artificial intelligence.")
        await cognee.cognify()

        results = await cognee.search(
            query_type=SearchType.GRAPH_COMPLETION,
            query_text="What is NLP?",
        )
        for item in results:
            print(item)
    finally:
        unsub_sync()
        unsub_async()


if __name__ == "__main__":
    asyncio.run(main())
