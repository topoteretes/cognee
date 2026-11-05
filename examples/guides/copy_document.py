"""Preview and optionally copy an existing cognified document within one store.

uv run python examples/guides/copy_document.py SOURCE_UUID DATA_UUID TARGET_UUID
Add --copy to write the reviewed version. This uses the configured default user's
SDK read/write permissions. Cloud adds sharing authority, Jev and approval above
this primitive. No source material is printed and no LLM is called by the copy.
"""

import argparse
import asyncio
from uuid import UUID

import cognee


async def main(args):
    preview = await cognee.datasets.copy_data(args.source, args.data, args.target, preview=True)
    print({key: preview[key] for key in ("source_fingerprint", "nodes", "edges", "vectors")})
    if args.copy:
        receipt = await cognee.datasets.copy_data(
            args.source,
            args.data,
            args.target,
            expected_source_fingerprint=preview["source_fingerprint"],
        )
        print(receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "data", "target"):
        parser.add_argument(name, type=UUID)
    parser.add_argument("--copy", action="store_true")
    asyncio.run(main(parser.parse_args()))
