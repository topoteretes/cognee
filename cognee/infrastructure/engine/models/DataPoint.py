import pickle
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone
from typing_extensions import TypedDict
from typing import Optional, Any, Dict, List


# Define metadata type
class MetaData(TypedDict):
    """
    Represent a metadata structure with type and index fields.
    """

    type: str
    index_fields: list[str]


# Updated DataPoint model with versioning and new fields
class DataPoint(BaseModel):
    """
    Model representing a data point with versioning and metadata support.

    Public methods include:
    - get_embeddable_data
    - get_embeddable_properties
    - get_embeddable_property_names
    - update_version
    - to_json
    - from_json
    - to_pickle
    - from_pickle
    - to_dict
    - from_dict
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

    @classmethod
    def get_embeddable_properties(self, data_point: "DataPoint"):
        """
        Retrieve a list of embeddable properties from the data point.

        This method returns a list of attribute values based on the index fields defined in the
        data point's metadata. If there are no index fields, it returns an empty list.

        Parameters:
        -----------

            - data_point ('DataPoint'): The DataPoint instance from which to retrieve embeddable
              properties.

        Returns:
        --------

            A list of embeddable property values, or an empty list if none exist.
        """
        if data_point.metadata and len(data_point.metadata["index_fields"]) > 0:
            return [
                getattr(data_point, field, None) for field in data_point.metadata["index_fields"]
            ]

        return []

    @classmethod
    def get_embeddable_property_names(self, data_point: "DataPoint"):
        """
        Retrieve the names of embeddable properties defined in the metadata.

        If no index fields are defined in the metadata, this method will return an empty list.

        Parameters:
        -----------

            - data_point ('DataPoint'): The DataPoint instance from which to retrieve embeddable
              property names.

        Returns:
        --------

            A list of property names corresponding to the index fields, or an empty list if none
            exist.
        """
        return data_point.metadata["index_fields"] or []

    def update_version(self):
        """
        Increment the version number of the data point and update the timestamp.

        This method will automatically modify the version attribute and refresh the updated_at
        timestamp to the current time.
        """
        self.version += 1
        self.updated_at = int(datetime.now(timezone.utc).timestamp() * 1000)

    # JSON Serialization
    def to_json(self) -> str:
        """
        Serialize the DataPoint instance to a JSON string format.

        This method uses the model's built-in serialization functionality to convert the
        instance into a JSON-compatible string.

        Returns:
        --------

            - str: The JSON string representation of the DataPoint instance.
        """
        return self.json()

    @classmethod
    def from_json(self, json_str: str):
        """
        Deserialize a DataPoint instance from a JSON string.

        The method transforms the input JSON string back into a DataPoint instance using model
        validation.

        Parameters:
        -----------

            - json_str (str): The JSON string representation of a DataPoint instance to be
              deserialized.

        Returns:
        --------

            A new DataPoint instance created from the JSON data.
        """
        return self.model_validate_json(json_str)

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        """
        Convert the DataPoint instance to a dictionary representation.

        This method uses the model's built-in functionality to serialize the instance attributes
        to a dictionary, which can optionally include additional arguments.

        Parameters:
        -----------

            - **kwargs: Additional keyword arguments for serialization options.

        Returns:
        --------

            - Dict[str, Any]: A dictionary representation of the DataPoint instance.
        """
        return self.model_dump(**kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataPoint":
        """
        Instantiate a DataPoint from a dictionary of attribute values.

        The method validates the incoming dictionary data against the model's schema and
        constructs a new DataPoint instance accordingly.

        Parameters:
        -----------

            - data (Dict[str, Any]): A dictionary containing the attributes of a DataPoint
              instance.

        Returns:
        --------

            - 'DataPoint': A new DataPoint instance constructed from the provided dictionary
              data.
        """
        return cls.model_validate(data)
