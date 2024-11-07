from uuid import UUID, uuid4
from typing import Optional
from pydantic import BaseModel
from .models.Task import Task

class PipelineConfig(BaseModel):
    batch_count: int = 10
    description: Optional[str] = None

class Pipeline():
    id: UUID = uuid4()
    name: str
    description: str
    tasks: list[Task] = []

    def __init__(self, name: str, pipeline_config: PipelineConfig):
        self.name = name
        self.description = pipeline_config.description
