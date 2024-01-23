from enum import Enum

class EmbeddingsType(Enum):
    OPEN_AI = 'open_ai'
    COHERE = 'cohere'
    SELF_HOSTED = 'self_hosted'
    HUGGING_FACE = 'hugging_face'
    IMAGE = 'image'