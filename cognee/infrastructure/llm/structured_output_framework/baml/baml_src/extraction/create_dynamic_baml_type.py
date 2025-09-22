from typing import Union
from enum import Enum
from datetime import datetime
from typing import get_origin, get_args

from baml_py.baml_py import ClassBuilder
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def create_dynamic_baml_type(tb, baml_model, pydantic_model):
    if pydantic_model is str:
        baml_model.add_property("text", tb.string())
        return tb

    def map_type(field_type, field_info):
        """
        Convert a Python / Pydantic type  ->  BAML TypeBuilder representation.
        """

        origin = get_origin(field_type)  # e.g. list[…]  ->  list
        args = get_args(field_type)  # e.g. list[int] -> (int,)

        # ------------------------------------------------------------------
        # 1. Optional / Union  ------------------------------------------------
        # ------------------------------------------------------------------
        if origin is Union:
            non_none_args = [t for t in args if t is not type(None)]

            # Optional[T]  ⇢  exactly (T, NoneType)
            if len(args) == 2 and len(non_none_args) == 1:
                return map_type(non_none_args[0], field_info).optional()

            # Plain Union[A, B, …]
            return tb.union(*(map_type(t, field_info) for t in args))

        # ------------------------------------------------------------------
        # 2. List / Sequence  -------------------------------------------------
        # ------------------------------------------------------------------
        if origin in (list,):
            (inner_type,) = args  # list has exactly one parameter
            return map_type(inner_type, field_info).list()

        # ------------------------------------------------------------------
        # 3. Dict / Map -------------------------------------------------------
        # ------------------------------------------------------------------
        def _is_enum_subclass(key_type) -> bool:
            """Guarded issubclass – returns False when tp is not a class."""
            return isinstance(key_type, type) and issubclass(key_type, Enum)

        if origin in (dict,):
            key_type, value_type = args or (str, object)

            if key_type is not str and not _is_enum_subclass(key_type):
                raise ValueError("BAML maps only allow 'str' or Enum subclasses as keys")

            return tb.map(
                map_type(key_type, field_info),  # mostly tb.string() or enum
                map_type(value_type, field_info),
            )

        # ------------------------------------------------------------------
        # 4. Enum -------------------------------------------------------------
        # ------------------------------------------------------------------
        if _is_enum_subclass(field_type):
            enum_builder = tb.add_enum(field_type.__name__)
            for member in field_type:
                enum_builder.add_value(member.name)
            return enum_builder.type()

        # ------------------------------------------------------------------
        # 5. Nested Pydantic model -------------------------------------------
        # ------------------------------------------------------------------
        from pydantic import BaseModel  # local import

        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            try:
                # Create nested class if it doesn't exist
                nested_class = tb.add_class(field_type.__name__)
                # Find dynamic types of nested class
                create_dynamic_baml_type(tb, nested_class, field_type)
            except ValueError:
                # If nested class already exists get it
                nested_class = tb._tb.class_(field_type.__name__)

            # Return nested class model
            if isinstance(nested_class, ClassBuilder):
                # Different nested_class objects have different syntax for type information
                # If nested class already exists type information can be found using the field method
                return nested_class.field()
            else:
                # If nested class was created type information can be found using type method
                return nested_class.type()

        primitive_map = {
            str: tb.string(),
            int: tb.int(),
            float: tb.float(),
            bool: tb.bool(),
            datetime: tb.string(),  # BAML has no native datetime
        }
        if field_type in primitive_map:
            return primitive_map[field_type]

        raise ValueError(f"Unsupported type for BAML mapping: {field_type}")

    fields = pydantic_model.model_fields

    # Add fields
    for field_name, field_info in fields.items():
        field_type = field_info.annotation
        baml_type = map_type(field_type, field_info)

        # Add property with type
        prop = baml_model.add_property(field_name, baml_type)

        # Add description if available
        if field_info.description:
            prop.description(field_info.description)

    return tb
