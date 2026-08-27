"""Type-hint regression test for GraphVectorStoreInterface (Finding 14, COG-6335
review).

delete_by_dataset_id lost its ``-> None`` return-type annotation when it
started returning a SourceRefRemovalResult, and never gained the replacement
-- unlike UnifiedStoreEngine.delete_by_dataset_id, the concrete implementation,
which already annotates its return type correctly.
"""

import importlib
import inspect

interface_module = importlib.import_module(
    "cognee.infrastructure.databases.unified.graph_vector_store_interface"
)
unified_store_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.unified.unified_store_engine"
)


def test_delete_by_dataset_id_return_annotation_matches_concrete_implementation():
    abstract_signature = inspect.signature(
        interface_module.GraphVectorStoreInterface.delete_by_dataset_id
    )
    concrete_signature = inspect.signature(
        unified_store_engine_module.UnifiedStoreEngine.delete_by_dataset_id
    )

    # unified_store_engine.py has ``from __future__ import annotations``
    # (stringifying every annotation, quotes and all); the interface module
    # does not, so its forward reference is the plain string itself. Strip
    # any extra quoting so both forms compare as referring to the same type.
    assert str(abstract_signature.return_annotation).strip("'\"") == "SourceRefRemovalResult"
    assert str(concrete_signature.return_annotation).strip("'\"") == "SourceRefRemovalResult"
