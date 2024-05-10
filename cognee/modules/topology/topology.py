
import os
import glob
from pydantic import BaseModel, create_model
from typing import Dict, Type, Any

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
from datetime import datetime

from cognee import config
from cognee.infrastructure import infrastructure_config
from infer_data_topology import infer_data_topology



# class UserLocation(BaseModel):
#     location_id: str
#     description: str
#     default_relationship: Relationship = Relationship(type = "located_in")
#
# class UserProperties(BaseModel):
#     custom_properties: Optional[Dict[str, Any]] = None
#     location: Optional[UserLocation] = None
#
# class DefaultGraphModel(BaseModel):
#     node_id: str
#     user_properties: UserProperties = UserProperties()
#     documents: List[Document] = []
#     default_fields: Optional[Dict[str, Any]] = {}
#     default_relationship: Relationship = Relationship(type = "has_properties")
#
class Relationship(BaseModel):
    type: str
    source: Optional[str] = None
    target: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None



class Document(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    default_relationship: Relationship = Field(default_factory=lambda: Relationship(type="belongs_to"))


class DirectoryModel(BaseModel):
    name: str
    path: str
    summary: str
    documents: List[Document] = []
    subdirectories: List['DirectoryModel'] = []
    default_relationship: Relationship = Field(default_factory=lambda: Relationship(type="belongs_to"))

DirectoryModel.update_forward_refs()

class RepositoryMetadata(BaseModel):
    name: str
    summary: str
    owner: str
    description: Optional[str] = None
    directories: List[DirectoryModel] = []
    documents: List[Document] = []
    default_relationship: Relationship = Field(default_factory=lambda: Relationship(type="belongs_to"))

class GitHubRepositoryModel(BaseModel):
    metadata: RepositoryMetadata
    root_directory: DirectoryModel

class TopologyEngine:
    def __init__(self):
        self.models: Dict[str, Type[BaseModel]] = {}

    async def infer(self, repository: str):

        path = infrastructure_config.get_config()["data_root_directory"]

        path = path +"/"+ str(repository)
        print(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such directory: {path}")

        file_structure = {}
        for filename in glob.glob(f"{path}/**", recursive=True):
            if os.path.isfile(filename):
                key = os.path.relpath(filename, start=path).replace(os.path.sep, "__")
                file_structure[key] = (str, ...)  # Assuming content as string for simplicity


        result = await infer_data_topology(str(file_structure), GitHubRepositoryModel)

        return result

    def load(self, model_name: str):
        return self.models.get(model_name)

    def extrapolate(self, model_name: str):
        # This method would be implementation-specific depending on what "extrapolate" means
        pass


if __name__ == "__main__":
    data_directory_path = os.path.abspath("../../../.data")
    print(data_directory_path)
    config.data_root_directory(data_directory_path)
    cognee_directory_path = os.path.abspath("../.cognee_system")
    config.system_root_directory(cognee_directory_path)
    async def main():
        engine = TopologyEngine()
        # model = engine.load("GitHubRepositoryModel")
        # if model is None:
        #     raise ValueError("Model not found")
        result = await engine.infer("example")
        print(result)

    import asyncio
    asyncio.run(main())
    # result = engine.extrapolate("GitHubRepositoryModel")
    # print(result)