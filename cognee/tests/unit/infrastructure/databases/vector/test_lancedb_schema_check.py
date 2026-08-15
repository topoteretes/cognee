"""The proactive LanceDB schema check must converge and must not read data.

Both properties were violated on a 100k-node store (COG-6185): every startup
rewrote 206k-row collections that the previous startup had just rewritten,
because the comparison included a nullability flag the storage layer never
reproduces — and each check materialized the whole table just to reach its
schema.
"""

import pyarrow as pa

from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter


def _payload(nullable: bool) -> pa.DataType:
    return pa.struct(
        [
            pa.field("id", pa.string(), nullable=nullable),
            pa.field("text", pa.string(), nullable=nullable),
            pa.field("version", pa.int64(), nullable=nullable),
        ]
    )


def test_nullability_does_not_make_identical_schemas_differ():
    """LanceDB stores every struct field nullable; a model field that is not
    Optional would otherwise never match what was written for it."""
    stored = LanceDBAdapter._normalize_arrow_type(_payload(nullable=True))
    declared = LanceDBAdapter._normalize_arrow_type(_payload(nullable=False))

    assert stored == declared


def test_field_names_still_distinguish_schemas():
    """The check must keep catching a genuinely different field set."""
    without_field = LanceDBAdapter._normalize_arrow_type(_payload(nullable=True))
    with_field = LanceDBAdapter._normalize_arrow_type(
        pa.struct(
            [
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("version", pa.int64()),
                pa.field("valid_to", pa.int64()),
            ]
        )
    )

    assert without_field != with_field


def test_field_types_still_distinguish_schemas():
    as_int = LanceDBAdapter._normalize_arrow_type(pa.struct([pa.field("version", pa.int64())]))
    as_string = LanceDBAdapter._normalize_arrow_type(pa.struct([pa.field("version", pa.string())]))

    assert as_int != as_string


def test_list_element_nullability_is_also_ignored():
    """Same reasoning one level down: element nullability is storage's choice."""
    strict = LanceDBAdapter._normalize_arrow_type(
        pa.list_(pa.field("item", pa.string(), nullable=False))
    )
    lenient = LanceDBAdapter._normalize_arrow_type(
        pa.list_(pa.field("item", pa.string(), nullable=True))
    )

    assert strict == lenient


def test_list_element_types_still_distinguish_schemas():
    of_strings = LanceDBAdapter._normalize_arrow_type(pa.list_(pa.string()))
    of_ints = LanceDBAdapter._normalize_arrow_type(pa.list_(pa.int64()))

    assert of_strings != of_ints
