import importlib
from unittest.mock import AsyncMock, patch

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.tasks.presort.default_spec import DEFAULT_PRESORT_SPEC
from cognee.tasks.presort.graph_apply import apply_presort_graph, build_graph_instances
from cognee.tasks.presort.models import FileRecord, PresortReport, RelationInstance

MODULE = "cognee.tasks.presort.graph_apply"

# The package re-exports a function named like its module, so patch the module
# object (a dotted target resolves to the function on Python 3.10).
run_custom_pipeline_module = importlib.import_module(
    "cognee.modules.run_custom_pipeline.run_custom_pipeline"
)


def _default_report():
    return PresortReport(
        scan_id="scan-1",
        root_path="/d/downloads",
        spec_used=DEFAULT_PRESORT_SPEC,
        files=[
            FileRecord(
                path="/d/a.pdf", name="a.pdf", extension="pdf", content_hash="h1", size_bytes=10
            ),
            FileRecord(
                path="/d/a (1).pdf",
                name="a (1).pdf",
                extension="pdf",
                content_hash="h1",
                size_bytes=10,
            ),
            FileRecord(
                path="/d/cv.txt", name="cv.txt", extension="txt", content_hash="h2", size_bytes=5
            ),
        ],
        relationships={
            "duplicate_of": [
                RelationInstance(
                    source="/d/a (1).pdf",
                    relation="duplicate_of",
                    target="/d/a.pdf",
                    target_entity="FileRecord",
                )
            ],
            "belongs_to_group": [
                RelationInstance(
                    source="/d/cv.txt",
                    relation="belongs_to_group",
                    target="docs",
                    target_entity="FileGroup",
                )
            ],
            "contains_pii": [
                RelationInstance(
                    source="/d/cv.txt",
                    relation="contains_pii",
                    target="resume",
                    target_entity="PiiTag",
                )
            ],
        },
    )


def test_build_graph_instances_from_default_spec():
    instances = build_graph_instances(_default_report())

    assert len(instances) == 3
    assert all(isinstance(instance, DataPoint) for instance in instances)
    by_name = {instance.name: instance for instance in instances}

    # Self-relation resolves to the sibling file instance (not a copy).
    duplicate = by_name["a (1).pdf"]
    assert duplicate.duplicate_of[0] is by_name["a.pdf"]

    # Cardinality one -> single assignment; many -> list.
    cv = by_name["cv.txt"]
    assert cv.belongs_to_group.name == "docs"
    assert [tag.name for tag in cv.contains_pii] == ["resume"]

    # Primitives declared on the spec are copied from the report records.
    assert by_name["a.pdf"].content_hash == "h1"
    assert by_name["a.pdf"].path == "/d/a.pdf"

    # identity_fields=["content_hash"] -> duplicate contents share one node id.
    assert by_name["a.pdf"].id == by_name["a (1).pdf"].id
    assert by_name["a.pdf"].id != cv.id


def test_build_graph_instances_custom_spec():
    spec = {
        "root": "Invoice",
        "entities": [
            {
                "name": "Invoice",
                "identity_fields": ["name"],
                "fields": [
                    {"kind": "primitive", "name": "path", "primitive_type": "string"},
                    {
                        "kind": "relation",
                        "name": "issued_by",
                        "relation": {"target_entity_name": "Vendor", "cardinality": "one"},
                    },
                ],
            },
            {"name": "Vendor", "identity_fields": ["name"]},
        ],
    }
    report = PresortReport(
        scan_id="s",
        root_path="/d",
        spec_used=spec,
        files=[FileRecord(path="/d/inv.pdf", name="inv.pdf", extension="pdf")],
        relationships={
            "issued_by": [
                RelationInstance(
                    source="/d/inv.pdf",
                    relation="issued_by",
                    target="ACME Corp",
                    target_entity="Vendor",
                    origin="llm",
                    confidence=0.9,
                )
            ]
        },
    )

    instances = build_graph_instances(report)

    assert len(instances) == 1
    invoice = instances[0]
    assert invoice.issued_by.name == "ACME Corp"
    assert type(invoice.issued_by).__name__ == "Vendor"


def test_unresolvable_relation_endpoints_skipped():
    report = _default_report()
    report.relationships["duplicate_of"].append(
        RelationInstance(
            source="/ghost.pdf",
            relation="duplicate_of",
            target="/d/a.pdf",
            target_entity="FileRecord",
        )
    )
    instances = build_graph_instances(report)  # must not raise
    assert len(instances) == 3


@pytest.mark.asyncio
async def test_apply_presort_graph_runs_custom_pipeline():
    report = _default_report()
    with patch.object(
        run_custom_pipeline_module,
        "run_custom_pipeline",
        new=AsyncMock(return_value="pipeline-info"),
    ) as pipeline_mock:
        result = await apply_presort_graph(report)

    pipeline_mock.assert_awaited_once()
    call_kwargs = pipeline_mock.await_args.kwargs
    assert call_kwargs["dataset"] == "downloads_presort_graph"
    assert call_kwargs["pipeline_name"] == "presort_graph_pipeline"
    assert len(call_kwargs["data"]) == 3  # all instances passed as the pipeline data
    assert result == {"dataset": "downloads_presort_graph", "nodes": 3, "result": "pipeline-info"}


@pytest.mark.asyncio
async def test_apply_presort_graph_empty_report():
    report = PresortReport(scan_id="s", root_path="/d", spec_used=DEFAULT_PRESORT_SPEC)
    assert await apply_presort_graph(report) is None
