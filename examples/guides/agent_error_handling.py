"""Branch on structured cognee errors — agent-facing demo (Pillar B #3604)."""

import asyncio

from cognee.exceptions import CogneeApiError, ErrorCode
from cognee.modules.retrieval.exceptions.exceptions import NoDataError


async def demo() -> None:
    scenarios = [
        NoDataError(stage="add"),
        NoDataError(stage="cognify"),
    ]

    for exc in scenarios:
        try:
            raise exc
        except CogneeApiError as error:
            envelope = error.to_dict()
            print(f"code={envelope['code']} retryable={envelope['retryable']}")
            print(f"remediation: {envelope['remediation']}")

            if envelope["code"] == ErrorCode.DATA_NOT_READY.value:
                stage = envelope.get("details", {}).get("stage")
                if stage == "cognify":
                    print("→ agent action: run cognify()")
                elif stage == "add":
                    print("→ agent action: run add()")
            print()


if __name__ == "__main__":
    asyncio.run(demo())
