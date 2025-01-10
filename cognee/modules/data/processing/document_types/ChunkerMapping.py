from cognee.modules.chunking.TextChunker import TextChunker


class ChunkerConfig:
    chunker_mapping = {"text_chunker": TextChunker}

    @classmethod
    def get_chunker(cls, chunker_name: str):
        chunker_class = cls.chunker_mapping.get(chunker_name)
        if chunker_class is None:
            raise NotImplementedError(
                f"Chunker '{chunker_name}' is not implemented. Available options: {list(cls.chunker_mapping.keys())}"
            )
        return chunker_class
