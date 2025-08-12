from uuid import UUID, uuid4
from typing import Optional, List
from typing_extensions import TypedDict
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


class MetaData(TypedDict):
    """
    Represent a metadata structure with type and index fields.
    """

    type: str
    index_fields: list[str]


class DataPoint(BaseModel):
    """
    Model representing a data point with versioning and metadata support.

    Public methods include:
    - get_embeddable_data
    - get_embeddable_properties
    - get_embeddable_property_names
    - update_version
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID = Field(default_factory=uuid4)
    created_at: int = Field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000)
    )
    updated_at: int = Field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000)
    )
    ontology_valid: bool = False
    version: int = 1  # Default version
    topological_rank: Optional[int] = 0
    metadata: Optional[MetaData] = {"index_fields": []}
    type: str = Field(default_factory=lambda: DataPoint.__name__)
    belongs_to_set: Optional[List["DataPoint"]] = None

    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, "type", self.__class__.__name__)

    @classmethod
    def get_embeddable_data(self, data_point: "DataPoint"):
        """
        Retrieve embeddable data from the data point object based on index fields.

        This method checks if there are defined index fields in the metadata and retrieves the
        value of the first indexed attribute. If the attribute is a string, it strips whitespace
        from it before returning.

        Parameters:
        -----------

            - data_point ('DataPoint'): The DataPoint instance from which to retrieve embeddable
              data.

        Returns:
        --------

            The value of the embeddable data, or None if not found.
        """
        if (
            data_point.metadata
            and len(data_point.metadata["index_fields"]) > 0
            and hasattr(data_point, data_point.metadata["index_fields"][0])
        ):
            attribute = getattr(data_point, data_point.metadata["index_fields"][0])

            if isinstance(attribute, str):
                return attribute.strip()
            return attribute

    def get_embeddable_properties(self):
        """
        Retrieve a list of embeddable properties from the data point.

        This method returns a list of attribute values based on the index fields defined in the
        data point's metadata. If there are no index fields, it returns an empty list.

        Returns:
        --------

            A list of embeddable property values, or an empty list if none exist.
        """
        if self.metadata and len(self.metadata["index_fields"]) > 0:
            return [getattr(self, field, None) for field in self.metadata["index_fields"]]

        return []

    def get_embeddable_property_names(self):
        """
        Retrieve the names of embeddable properties defined in the metadata.

        If no index fields are defined in the metadata, this method will return an empty list.

        Returns:
        --------

            A list of property names corresponding to the index fields, or an empty list if none
            exist.
        """
        if self.metadata:
            return self.metadata["index_fields"] or []

        return []

    def update_version(self):
        """
        Increment the version number of the data point and update the timestamp.

        This method will automatically modify the version attribute and refresh the updated_at
        timestamp to the current time.
        """
        self.version += 1
        self.updated_at = int(datetime.now(timezone.utc).timestamp() * 1000)
