from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.graph_models import GraphSchemaSpec
from cognee.tasks.presort.default_spec import DEFAULT_PRESORT_SPEC
from cognee.tasks.presort.models import (
    DuplicateCluster,
    FileRecord,
    PiiFinding,
    ProposedGroup,
    RelationInstance,
    VersionCandidate,
)
from cognee.tasks.presort.relations import (
    ExtractedRelationTarget,
    RelationExtraction,
    compute_relationships,
    register_relation_detector,
    unregister_relation_detector,
)

MODULE = "cognee.tasks.presort.relations"


def _default_inputs():
    files = [
        FileRecord(path="/d/a.pdf", name="a.pdf", extension="pdf"),
        FileRecord(path="/d/a (1).pdf", name="a (1).pdf", extension="pdf"),
        FileRecord(path="/d/a_v2.pdf", name="a_v2.pdf", extension="pdf"),
    ]
    duplicates = [DuplicateCluster(content_hash="h", paths=["/d/a.pdf", "/d/a (1).pdf"])]
    versions = [
        VersionCandidate(
            normalized_stem="a", extension="pdf", directory="/d", paths=["/d/a.pdf", "/d/a_v2.pdf"]
        )
    ]
    pii = [PiiFinding(path="/d/a.pdf", category="email_address", severity="low")]
    groups = [ProposedGroup(name="docs", dataset_name="docs", file_paths=["/d/a.pdf"])]
    return files, duplicates, versions, pii, groups


@pytest.mark.asyncio
async def test_builtin_relations_derived_from_sections():
    spec = GraphSchemaSpec.model_validate(DEFAULT_PRESORT_SPEC)
    files, duplicates, versions, pii, groups = _default_inputs()

    relationships, warnings = await compute_relationships(
        Path("/d"), spec, files, duplicates, versions, pii, groups
    )

    assert warnings == []
    assert set(relationships) == {"duplicate_of", "version_of", "belongs_to_group", "contains_pii"}

    dup = relationships["duplicate_of"][0]
    assert (dup.source, dup.target, dup.target_entity) == ("/d/a (1).pdf", "/d/a.pdf", "FileRecord")

    version = relationships["version_of"][0]
    assert (version.source, version.target) == ("/d/a.pdf", "/d/a_v2.pdf")

    group = relationships["belongs_to_group"][0]
    assert (group.target, group.target_entity) == ("docs", "FileGroup")

    tag = relationships["contains_pii"][0]
    assert (tag.target, tag.target_entity) == ("email_address", "PiiTag")


def _spec_with_custom_relation(relation_name="invoice_for"):
    return GraphSchemaSpec.model_validate(
        {
            "root": "FileRecord",
            "entities": [
                {
                    "name": "FileRecord",
                    "fields": [
                        {
                            "kind": "relation",
                            "name": relation_name,
                            "description": "The vendor this invoice was issued by.",
                            "relation": {"target_entity_name": "Vendor", "cardinality": "one"},
                        },
                    ],
                },
                {"name": "Vendor", "description": "A company issuing invoices."},
            ],
        }
    )


@pytest.mark.asyncio
async def test_unknown_relation_without_llm_warns():
    files, duplicates, versions, pii, groups = _default_inputs()

    relationships, warnings = await compute_relationships(
        Path("/d"), _spec_with_custom_relation(), files, duplicates, versions, pii, groups
    )

    assert relationships["invoice_for"] == []
    assert len(warnings) == 1
    assert "invoice_for" in warnings[0] and "no built-in" in warnings[0]


@pytest.mark.asyncio
async def test_registered_custom_detector_wins():
    files, duplicates, versions, pii, groups = _default_inputs()

    def detector(ctx):
        return [
            RelationInstance(
                source=ctx.files[0].path,
                relation=ctx.relation.name,
                target="ACME Corp",
                target_entity="Vendor",
            )
        ]

    register_relation_detector("invoice_for", detector)
    try:
        relationships, warnings = await compute_relationships(
            Path("/d"), _spec_with_custom_relation(), files, duplicates, versions, pii, groups
        )
    finally:
        assert unregister_relation_detector("invoice_for") is detector

    assert warnings == []
    instance = relationships["invoice_for"][0]
    assert instance.target == "ACME Corp"
    assert instance.origin == "custom"


@pytest.mark.asyncio
async def test_async_custom_detector_supported():
    files, duplicates, versions, pii, groups = _default_inputs()

    async def detector(ctx):
        return [
            RelationInstance(
                source="/d/a.pdf", relation=ctx.relation.name, target="X", target_entity="Vendor"
            )
        ]

    register_relation_detector("invoice_for", detector)
    try:
        relationships, _ = await compute_relationships(
            Path("/d"), _spec_with_custom_relation(), files, duplicates, versions, pii, groups
        )
    finally:
        unregister_relation_detector("invoice_for")

    assert relationships["invoice_for"][0].target == "X"


@pytest.mark.asyncio
async def test_llm_fallback_for_unknown_relation(tmp_path):
    invoice = tmp_path / "invoice_march.txt"
    invoice.write_text("Invoice issued by ACME Corp for March services.")
    files = [
        FileRecord(path=str(invoice), name="invoice_march.txt", extension="txt", is_text=True),
        FileRecord(path="/d/photo.jpg", name="photo.jpg", extension="jpg", is_text=False),
    ]

    extraction = RelationExtraction(
        instances=[
            ExtractedRelationTarget(
                target_name="ACME Corp", confidence=0.9, rationale="issuer named in header"
            ),
            ExtractedRelationTarget(
                target_name="Maybe Inc", confidence=0.2, rationale="weak mention"
            ),
        ]
    )
    with patch(
        f"{MODULE}.LLMGateway.acreate_structured_output",
        new=AsyncMock(return_value=extraction),
    ) as llm_mock:
        relationships, warnings = await compute_relationships(
            Path(tmp_path),
            _spec_with_custom_relation(),
            files,
            [],
            [],
            [],
            [],
            use_llm=True,
        )

    llm_mock.assert_awaited_once()  # only the text file is analyzed
    assert warnings == []
    instances = relationships["invoice_for"]
    assert len(instances) == 1  # low-confidence hit filtered out
    assert instances[0].target == "ACME Corp"
    assert instances[0].origin == "llm"
    assert instances[0].confidence == 0.9
