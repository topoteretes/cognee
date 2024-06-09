from typing import Dict

from cognee.infrastructure.data.chunking.LangchainChunkingEngine import LangchainChunkEngine


class ChunkingConfig(Dict):
    vector_db_url: str
    vector_db_key: str
    vector_db_provider: str

def create_chunking_engine(config: ChunkingConfig):
    if config["chunk_engine"] == "langchainchunkengine":

        return LangchainChunkEngine(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            chunk_strategy=config["chunk_strategy"],


        )
    elif config["chunk_engine"] == "defaultchunkengine":
            from .DefaultChunkEngine import DefaultChunkEngine

            return DefaultChunkEngine(
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"],
                chunk_strategy=config["chunk_strategy"],
            )
    elif config["chunk_engine"] == "haystackchunkengine":
        from .HaystackChunkEngine import HaystackChunkEngine

        return HaystackChunkEngine(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            chunk_strategy=config["chunk_strategy"],
        )
