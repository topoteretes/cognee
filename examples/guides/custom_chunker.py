"""Cut documents with your own chunker, and keep update() chunk-level.

A deployment registers a chunker under a name and names it in DEFAULT_CHUNKER.
cognify(), remember() and update() then cut with it, from the SDK and from the API.

SectionChunker cuts a Markdown document at its headings, so a section never shares
a chunk with the next one. A section over the token budget is cut further by
TextChunker. It has what update() needs to replace only the chunks an edit touched
(the full list is in cognee/modules/chunking/chunker_registry.py):
- its own chunker_id, which cognee stamps on every chunk it cuts;
- chunks that join back to the document's text, with no overlap.

Editing one section below re-extracts that section only: update() reports
"incremental" with one region. A document cut by another chunker, for example one
stored before DEFAULT_CHUNKER changed, is rebuilt once ("full_rebuild", reason
"unsupported_chunker") and is chunk-level again from its next update.

Requires: LLM_API_KEY.
Run: uv run python examples/guides/custom_chunker.py
"""

import asyncio
import os
import re

# A deployment sets this in its environment or .env.
os.environ.setdefault("DEFAULT_CHUNKER", "SectionChunker")

import cognee
from cognee.modules.chunking.chunk_id import content_chunk_id
from cognee.modules.chunking.chunker_registry import register_chunker
from cognee.modules.chunking.TextChunker import TextChunker

DATASET = "custom_chunker_guide"

HANDBOOK = """# Opening hours

The Quillon office opens at nine and closes at six on weekdays.

# Visitors

Visitors sign in at the front desk and are met by Ingrid Solvay.

# Deliveries

Deliveries are taken at the side entrance by Tomas Ferreira until noon.
"""


class SectionChunker(TextChunker):
    """One Markdown section per chunk; TextChunker cuts a section over the budget."""

    chunker_id = "section_chunker_v1"

    async def read(self):
        document_id = str(self.document.id)
        occurrences: dict[str, int] = {}
        async for text in self.get_text():
            # Split before each heading line. The pieces join back to ``text``.
            for section in re.split(r"(?m)^(?=#)", text):
                if not section:
                    continue

                async def section_text(section=section):
                    yield section

                section_chunker = TextChunker(self.document, section_text, self.max_chunk_size)
                async for chunk in section_chunker.read():
                    # Chunk ids count repeated texts over the document, not one section.
                    occurrence = occurrences.get(chunk.content_hash, 0)
                    occurrences[chunk.content_hash] = occurrence + 1
                    chunk.id = content_chunk_id(document_id, chunk.content_hash, occurrence)
                    chunk.chunker_id = self.chunker_id
                    chunk.chunk_index = self.chunk_index
                    self.chunk_index += 1
                    yield chunk


async def main():
    # Register before the first cognify call; DEFAULT_CHUNKER names it.
    register_chunker("SectionChunker", SectionChunker)

    # Start clean (optional in your app)
    await cognee.forget(everything=True)

    await cognee.remember(HANDBOOK, dataset_name=DATASET, self_improvement=False)

    dataset = next(d for d in await cognee.datasets.list_datasets() if d.name == DATASET)
    document = (await cognee.datasets.list_data(dataset.id))[0]

    # Edit one section. Only its chunk is cut and extracted again.
    edited = HANDBOOK.replace("closes at six", "closes at eight")
    result = await cognee.update(data_id=document.id, data=edited, dataset_id=dataset.id)
    print(
        f"update: {result['status']}, regions={result['regions']}, "
        f"kept={result['kept_chunks']}, added={result['added_chunks']}, "
        f"deleted={result['deleted_chunks']}"
    )

    answers = await cognee.recall("When does the Quillon office close?", datasets=[DATASET])
    print(answers[0].text if answers else "(nothing found)")


if __name__ == "__main__":
    asyncio.run(main())
