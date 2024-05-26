import os
import glob
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Type, Any, Tuple
from datetime import datetime

from cognee import config
from cognee.base_config import get_base_config
from cognee.infrastructure import infrastructure_config
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.topology.infer_data_topology import infer_data_topology
cognify_config = get_cognify_config()
base_config = get_base_config()

class Relationship(BaseModel):
    type: str = Field(..., description="The type of relationship, e.g., 'belongs_to'.")
    source: Optional[str] = Field(None, description="The identifier of the source id of in the relationship being a directory or subdirectory")
    target: Optional[str] = Field(None, description="The identifier of the target id in the relationship being the directory, subdirectory or file")
    properties: Optional[Dict[str, Any]] = Field(None, description="A dictionary of additional properties and values related to the relationship.")


class Document(BaseModel):
    node_id: str
    title: str
    description: Optional[str] = None
    default_relationship: Relationship


class DirectoryModel(BaseModel):
    node_id: str
    path: str
    summary: str
    documents: List[Document] = []
    subdirectories: List['DirectoryModel'] = []
    default_relationship: Relationship


DirectoryModel.update_forward_refs()


class DirMetadata(BaseModel):
    node_id: str
    summary: str
    owner: str
    description: Optional[str] = None
    directories: List[DirectoryModel] = []
    documents: List[Document] = []
    default_relationship: Relationship


class GitHubRepositoryModel(BaseModel):
    node_id: str
    metadata: DirMetadata
    root_directory: DirectoryModel


class TopologyEngine:
    def __init__(self) -> None:
        self.models: Dict[str, Type[BaseModel]] = {}

    async def populate_model(self, directory_path: str, file_structure: Dict[str, Union[Dict, Tuple[str, ...]]], parent_id: Optional[str] = None) -> DirectoryModel:
        directory_id = os.path.basename(directory_path) or "root"
        directory = DirectoryModel(
            node_id=directory_id,
            path=directory_path,
            summary=f"Contents of {directory_id}",
            default_relationship=Relationship(type="contains", source=parent_id, target=directory_id)
        )

        for key, value in file_structure.items():
            if isinstance(value, dict):
                # Recurse into subdirectory
                subdirectory_path = os.path.join(directory_path, key)
                subdirectory = await self.populate_model(subdirectory_path, value, parent_id=directory_id)
                directory.subdirectories.append(subdirectory)
            elif isinstance(value, tuple) and value[0] == 'file':
                # Handle file
                document = Document(
                    node_id=key,
                    title=key,
                    default_relationship=Relationship(type="contained_by", source=key, target=directory_id)
                )
                directory.documents.append(document)

        return directory

    async def infer_from_directory_structure(self, node_id: str, repository: str, model: Type[BaseModel]) -> GitHubRepositoryModel:
        """ Infer the topology of a repository from its file structure """

        path = base_config.data_root_directory
        path = path + "/" + str(repository)
        print(path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"No such directory: {path}")

        root: Dict[str, Union[Dict, Tuple[str, ...]]] = {}
        for filename in glob.glob(f"{path}/**", recursive=True):
            parts = os.path.relpath(filename, start=path).split(os.path.sep)
            current = root
            for part in parts[:-1]:  # Traverse/create to the last directory
                if part not in current:
                    current[part] = {}
                current = current[part]
            last_part = parts[-1]
            if os.path.isfile(filename):
                current[last_part] = ("file", ...)  # Placeholder for file content or metadata
            elif os.path.isdir(filename):
                if last_part not in current:  # Only create a new directory entry if it doesn't exist
                    current[last_part] = {}

        root_directory = await self.populate_model('/', root)

        repository_metadata = DirMetadata(
            node_id="repo1",
            summary="Example repository",
            owner="user1",
            directories=[root_directory],
            documents=[],
            default_relationship=Relationship(type="contained_by", source="repo1", target=node_id)
        )

        active_model = GitHubRepositoryModel(
            node_id=node_id,
            metadata=repository_metadata,
            root_directory=root_directory
        )

        return active_model

    def load(self, model_name: str) -> Optional[Type[BaseModel]]:
        return self.models.get(model_name)

    def extrapolate(self, model_name: str) -> None:
        # This method would be implementation-specific depending on what "extrapolate" means
        pass


if __name__ == "__main__":
    data_directory_path = os.path.abspath("../../../.data")
    print(data_directory_path)
    config.data_root_directory(data_directory_path)
    cognee_directory_path = os.path.abspath("../.cognee_system")
    config.system_root_directory(cognee_directory_path)

    async def main() -> None:
        engine = TopologyEngine()
        # model = engine.load("GitHubRepositoryModel")
        # if model is None:
        #     raise ValueError("Model not found")
        result = await engine.infer_from_directory_structure("example_node_id", "example_repo", GitHubRepositoryModel)
        print(result)

    import asyncio
    asyncio.run(main())
    # result = engine.extrapolate("GitHubRepositoryModel")
    # print(result)
