import logging
from datetime import datetime, timezone
from typing import Any
from uuid import NAMESPACE_OID, UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired, TypedDict

from cognee.infrastructure.engine.models.FieldAnnotations import _Dedup, _Embeddable
from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id

logger = logging.getLogger(__name__)


# Define metadata type
class MetaData(TypedDict):
    """
    Represent a metadata structure with type and index fields.
    """

    type: NotRequired[str]
    index_fields: list[str]
    identity_fields: NotRequired[list[str]]


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
    - to_dict
    - from_dict
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Defaults to a random UUID. A random id has NO stable identity, so such a
    # node never deduplicates/merges across runs or mentions and cannot be looked
    # up by recomputing its id. For a node that should be mergeable/idempotent
    # (like Entity), declare ``identity_fields`` in ``metadata`` (or pass an
    # explicit id via ``id_for``); the id is then derived deterministically from
    # those fields, namespaced by class name. Custom user models must opt in.
    id: UUID = Field(default_factory=uuid4)
    created_at: int = Field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000)
    )
    updated_at: int = Field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000)
    )
    ontology_valid: bool = False
    version: int = 1  # Default version
    topological_rank: int | None = 0
    metadata: MetaData = {"index_fields": []}
    type: str = Field(default_factory=lambda: DataPoint.__name__)
    belongs_to_set: "list[DataPoint] | list[str] | None" = None
    source_pipeline: str | None = None
    source_task: str | None = None
    source_node_set: str | None = None
    source_user: str | None = None
    source_content_hash: str | None = None
    feedback_weight: float = 0.5
    # A never-recalled node has no usage boost; first recall initializes it from zero.
    frequency_weight: float = 0.0
    importance_weight: float | None = 0.5

    def __init__(self, **data: Any) -> None:
        explicit_id = "id" in data
        super().__init__(**data)
        object.__setattr__(self, "type", self.__class__.__name__)
        if not explicit_id:
            identity_fields = self.__class__._get_identity_fields()
            if identity_fields:
                # self.__dict__ holds the validated field values (defaults
                # applied) — no model_dump(): a full recursive serialization on
                # every construction is pure waste for reading 1-2 fields.
                identity_id = self.__class__._generate_identity_id(identity_fields, self.__dict__)
                if identity_id is not None:
                    object.__setattr__(self, "id", identity_id)

    @classmethod
    def _get_identity_fields(cls) -> list[str] | None:
        """Get identity_fields from the class's metadata field default, if defined.

        Walks the MRO to detect if a parent class defined identity_fields that a
        subclass accidentally dropped when overriding metadata.
        """
        metadata_field = cls.model_fields.get("metadata")
        if metadata_field is not None and metadata_field.default is not None:
            identity = metadata_field.default.get("identity_fields")
            if identity is None:
                for parent in cls.__mro__[1:]:
                    parent_meta = getattr(parent, "model_fields", {}).get("metadata")
                    if parent_meta is not None and parent_meta.default is not None:
                        parent_identity = parent_meta.default.get("identity_fields")
                        if parent_identity is not None:
                            logger.warning(
                                "%s overrides metadata but drops identity_fields "
                                "defined in parent %s",
                                cls.__name__,
                                parent.__name__,
                            )
                            break
            return identity
        return None

    @classmethod
    def _generate_identity_id(cls, identity_fields: list[str], data: dict) -> UUID | None:
        """Generate the deterministic id of an instance from its ``identity_fields``.

        Collects the identity field values (from ``data`` or, if absent there, the
        Pydantic field default) and delegates the actual id derivation to
        :meth:`id_for`. This is intentional: ``id_for`` is the single source of
        truth for id creation, so an instance built without an explicit id and a
        bare ``Model.id_for(...)`` lookup can never drift apart.

        Returns ``None`` if any identity field is missing from both ``data`` and
        the Pydantic field defaults, which makes ``__init__`` fall back to the
        default UUID4.
        """
        values = []
        for field_name in identity_fields:
            if field_name in data:
                values.append(data[field_name])
            else:
                # Field absent from the instance values (e.g. references a
                # non-existent attribute) — fall back to its Pydantic default,
                # or bail out.
                field_info = cls.model_fields.get(field_name)
                if field_info is not None and field_info.default is not None:
                    values.append(field_info.default)
                else:
                    return None
        return cls.id_for(*values)

    @staticmethod
    def _normalize_identity_value(value: Any) -> str:
        """Normalize a single identity value (lower-case, spaces→_, strip apostrophes).

        Kept byte-for-byte aligned with ``generate_node_id`` (the legacy bare-name
        hashing) so historical ids remain recomputable from a normalized value —
        the graph id migration relies on this. Pinned by
        ``test_identity_fields.py::TestNormalizationMatchesGenerateNodeId``.
        """
        if isinstance(value, str):
            return value.lower().replace(" ", "_").replace("'", "")
        return str(value)

    @classmethod
    def id_for(cls, *values: Any) -> UUID:
        """Return the deterministic node id for this model from its identity value(s).

        The id namespace is the class name itself —
        ``uuid5(NAMESPACE_OID, f"{cls.__name__}:{values}")`` — so two different node
        types can never collide on the same input string, and callers cannot forget
        or mistype a namespace prefix: the class supplies it. This is the single
        source of truth for "what id does a node of this kind with this identity
        have", used both when creating nodes and when looking them up from a raw
        string before an instance exists.

        ``_generate_identity_id`` (the ``identity_fields`` path in ``__init__``)
        delegates here, so an instance's auto-derived id and ``Model.id_for(...)``
        are guaranteed to be the same value for the same inputs.
        """
        joined = "|".join(cls._normalize_identity_value(value) for value in values)
        return uuid5(NAMESPACE_OID, f"{cls.__name__}:{joined}")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-derive metadata index_fields and identity_fields from Annotated markers.

        If a subclass uses Annotated[str, Embeddable()] or Annotated[str, Dedup()]
        on its fields, and does NOT explicitly set metadata, the metadata default
        is automatically populated from those annotations.
        """
        super().__pydantic_init_subclass__(**kwargs)

        # Only auto-derive if the subclass didn't explicitly declare metadata
        if "metadata" in cls.__annotations__:
            return

        embeddable_fields = []
        dedup_fields = []

        for field_name, field_info in cls.model_fields.items():
            if field_info.metadata:
                for meta in field_info.metadata:
                    if isinstance(meta, _Embeddable):
                        embeddable_fields.append(field_name)
                    if isinstance(meta, _Dedup):
                        dedup_fields.append(field_name)

        if embeddable_fields or dedup_fields:
            new_metadata = {"index_fields": embeddable_fields}
            if dedup_fields:
                new_metadata["identity_fields"] = dedup_fields
            cls.model_fields["metadata"].default = new_metadata

    @classmethod
    def get_embeddable_data(cls, data_point: "DataPoint") -> Any | None:
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
    def get_embeddable_properties(cls, data_point: "DataPoint") -> list[Any | None]:
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
    def get_embeddable_property_names(cls, data_point: "DataPoint") -> list[str]:
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

    def update_version(self) -> None:
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
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "DataPoint":
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
        return cls.model_validate_json(json_str)

    def to_dict(self, **kwargs: Any) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> "DataPoint":
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
