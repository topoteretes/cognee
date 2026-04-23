"""Chunker that preserves BEAM conversation structure and factual fidelity."""

from uuid import NAMESPACE_OID, uuid5

from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.conversation_preprocessing import (
    build_preprocessed_fragments_from_text,
)
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk


class ConversationChunker(Chunker):
    """Chunk conversation transcripts with BEAM-aware turn pairing."""

    async def read(self):
        full_text = ""
        async for content_text in self.get_text():
            full_text += content_text

        chunk_count = 0
        for idx, fragment in enumerate(
            build_preprocessed_fragments_from_text(full_text, self.max_chunk_size)
        ):
            yield DocumentChunk(
                id=uuid5(NAMESPACE_OID, f"{str(self.document.id)}-{idx}"),
                text=fragment.text,
                chunk_size=fragment.chunk_size,
                is_part_of=self.document,
                chunk_index=idx,
                cut_type="conversation_turn_pair"
                if fragment.pair_complete
                else "conversation_turn",
                contains=[],
                metadata={
                    "index_fields": ["text"],
                    "session": fragment.session,
                    "turn": fragment.turn,
                    "turn_status": fragment.turn_status,
                    "pair_complete": fragment.pair_complete,
                    "part": fragment.part,
                    "part_count": fragment.part_count,
                    "fragment_kind": fragment.fragment_kind,
                },
            )
            chunk_count = idx + 1

        self.chunk_index = chunk_count
