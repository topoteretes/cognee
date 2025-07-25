from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from cognee.shared.data_models import DefaultContentPrediction, SummarizedContent


class CognifyConfig(BaseSettings):
    classification_model: object = DefaultContentPrediction
    summarization_model: object = SummarizedContent
    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_cognify_config():
    return CognifyConfig()
