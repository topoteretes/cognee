from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.root_dir import get_absolute_path

from cognee.shared.data_models import MonitoringTool, DefaultContentPrediction, LabeledContent, SummarizedContent, \
    DefaultCognitiveLayer


# Monitoring tool



class CognifyConfig(BaseSettings):
    classification_model: object = DefaultContentPrediction
    summarization_model: object = SummarizedContent
    labeling_model: object = LabeledContent
    cognitive_layer_model: object = DefaultCognitiveLayer
    intra_layer_score_treshold: int = 0.98
    connect_documents: bool = False



    model_config = SettingsConfigDict(env_file = ".env", extra = "allow")

    def to_dict(self) -> dict:
        return {
            "classification_model": self.classification_model,
            "summarization_model": self.summarization_model,
            "labeling_model": self.labeling_model,
            "cognitive_layer_model": self.cognitive_layer_model,
            "intra_layer_score_treshold": self.intra_layer_score_treshold,
            "connect_documents": self.connect_documents,
        }

@lru_cache
def get_cognify_config():
    return CognifyConfig()