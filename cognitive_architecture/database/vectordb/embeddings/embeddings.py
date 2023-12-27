from .embeddings_type import EmbeddingsType
from ..chunkers.chunk_strategy import ChunkStrategy


class Embeddings:
    def __init__(self, embeddings_type: EmbeddingsType = EmbeddingsType.OPEN_AI,
                 chunk_size: int = 256,
                 chunk_overlap: int = 128,
                 chunk_strategy: ChunkStrategy = ChunkStrategy.EXACT,
                 docker_image: str = None,
                 hugging_face_model_name: str = None):
        self.embeddings_type = embeddings_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
        self.docker_image = docker_image
        self.hugging_face_model_name = hugging_face_model_name

    def serialize(self):
        data = {
            'embeddings_type': self.embeddings_type.name if self.embeddings_type else None,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'chunk_strategy': self.chunk_strategy.name if self.chunk_strategy else None,
            'docker_image': self.docker_image,
            'hugging_face_model_name': self.hugging_face_model_name
        }

        return {k: v for k, v in data.items() if v is not None}
