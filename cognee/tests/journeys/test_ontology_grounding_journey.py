"""Recall names the authoritative table for an ontology term, on real stores.

A database source is cognified with an ontology configured: its ``customers`` table
realizes the ontology class ``Customer`` and is listed as the system of record. A
question that says "customer" must then reach that class node through grounding,
and the context the LLM would receive must explain the term and name the table with
its owner. Unit tests mock the stores; this one proves the node ids and the edge
properties survive Ladybug and LanceDB.
"""

import sqlite3

import pytest

import cognee
from cognee.modules.ontology.ontology_env_config import get_ontology_env_config
from cognee.modules.search.types import SearchType

ONTOLOGY = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
  <owl:Class rdf:about="http://example.org/business#Party"/>
  <owl:Class rdf:about="http://example.org/business#Customer">
    <rdfs:subClassOf rdf:resource="http://example.org/business#Party"/>
  </owl:Class>
</rdf:RDF>
"""


@pytest.fixture
def ontology_configured(tmp_path, monkeypatch):
    """ONTOLOGY_FILE_PATH and ONTOLOGY_AUTHORITATIVE_SOURCES set for this test only."""
    ontology = tmp_path / "business.owl"
    ontology.write_text(ONTOLOGY)
    monkeypatch.setenv("ONTOLOGY_FILE_PATH", str(ontology))
    # The source below is the SQLite file ``crm``, so its table is ``crm.customers``.
    monkeypatch.setenv("ONTOLOGY_AUTHORITATIVE_SOURCES", "crm.customers=sales-ops")
    get_ontology_env_config.cache_clear()
    yield
    get_ontology_env_config.cache_clear()


@pytest.mark.journey
@pytest.mark.asyncio
async def test_recall_names_the_authoritative_table_for_an_ontology_term(
    clean_env, default_user, tmp_path, monkeypatch, ontology_configured
):
    pytest.importorskip("dlt")
    monkeypatch.setenv("DLT_DATA_DIR", str(tmp_path / "dlt"))
    source = tmp_path / "crm"
    with sqlite3.connect(source) as connection:
        connection.executescript(
            "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);"
            "INSERT INTO customers VALUES (1, 'Acme Corporation');"
        )

    await cognee.add(f"sqlite:///{source}", dataset_name="crm_journey")
    await cognee.cognify(datasets=["crm_journey"])

    results = await cognee.recall(
        "which table holds customer data?",
        query_type=SearchType.HYBRID_COMPLETION,
        datasets=["crm_journey"],
        only_context=True,
        user=default_user,
    )

    context = " ".join(result.text for result in results)
    assert "## Ontology grounding" in context
    assert '"customer" refers to Customer (class); is a Party' in context
    assert "customers -- realizes -- customer (authoritative source, owner: sales-ops)" in context
