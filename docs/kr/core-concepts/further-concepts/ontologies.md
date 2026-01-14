# Ontologies

> Enrich your knowledge graph with external vocabularies

## What is an ontology in Cognee?

An **ontology** is an optional RDF/OWL file you can provide to Cognee.
It acts as a **reference vocabulary**, making sure that entity types ("classes") and entity mentions ("individuals") extracted from your data are linked to canonical, well-defined concepts.

## How it works

* You pass `ontology_file_path="my_ontology.owl"` when running [Cognify](../main-operations/cognify).
* Cognee parses the file with [RDFLib](https://rdflib.dev/) and loads its classes and relationships.
* During graph extraction, entities and types are checked against the ontology:
  * If a match is found, the node is marked `ontology_valid=True`.
  * Parent classes and object-property links from the ontology are attached as extra edges.
* If no ontology is provided, extraction still works, just without validation or enrichment.

## Why use an ontology

* **Consistency**: standardize how entities and types are represented
* **Enrichment**: bring in inherited relationships from a domain schema
* **Control**: align Cognee's graph with existing enterprise or scientific vocabularies

## Where to get ontologies

Ontologies are an art and science on their own. Cognee works best with **manually curated, focused ontologies** that fit your dataset. The simplest way to start is to **create a small ontology yourself** — just a few classes and relationships that match the entities you expect.

Public resources like **Wikidata** or **DBpedia** define millions of classes and entities, which makes them too big to use directly in Cognee. If you are not creating an ontology from scratch, you can start from a public one — but always work with a subset, not the full ontology:

* **Select only the pieces you need** (specific classes, properties, or individuals)
* **Save the subset** in a format Cognee can parse with [`rdflib`](https://rdflib.readthedocs.io/)
* **If needed, enrich the subset manually** by adding extra classes or relationships relevant to your domain
* **Keep it small and relevant** so matching stays precise and performance remains fast

<AccordionGroup>
  <Accordion title="Common sources">
    - **General vocabularies**: schema.org, Dublin Core Terms (DC/Terms), SKOS, PROV-O, FOAF
    - **Knowledge graph backbones**: DBpedia Ontology, Wikidata (Wikibase RDF ontology)
    - **Domain examples**:
      * Healthcare: SNOMED CT (licensed), ICD, UMLS, MeSH, HL7/FHIR RDF
      * Finance: FIBO (Financial Industry Business Ontology)
      * Geo/IoT: GeoSPARQL, SOSA/SSN, GeoNames
      * Units: QUDT
  </Accordion>

  <Accordion title="Why subsetting is essential">
    Every public ontology is **too broad to ingest wholesale**. Creating a subset is what makes them usable in Cognee:

    * Improves matching precision (fewer false matches when mapping LLM output)
    * Keeps performance acceptable (smaller graphs → faster resolution)
    * Lets you curate only the relevant parts of a domain
  </Accordion>

  <Accordion title="How subsetting works">
    Different communities provide different ways to extract subsets (e.g., "slims" in OBO ontologies, WDumper for Wikidata, module extraction in Protégé). The details vary, but the general principle is the same:

    1. Pick the terms (classes or properties) you care about
    2. Extract those terms plus their immediate context (e.g. parent classes, related properties)
    3. Save the result in an `rdflib`-readable RDF format
  </Accordion>
</AccordionGroup>

## Supported formats

Any format [RDFLib](https://rdflib.readthedocs.io/) can parse:

* RDF/XML (`.owl`, `.rdf`)
* Turtle (`.ttl`)
* N-Triples, JSON-LD, and others

## Practical example

Once you have your subset file, integrating it into Cognee is simple:

```python  theme={null}
import cognee

await cognee.cognify(
    datasets=["my_dataset"],
    ontology_file_path="subset.owl",  # your curated subset here
)
```

For more detailed examples of working with ontologies in Cognee, check out the demo scripts in the repository:

* [Basic ontology demo](https://github.com/topoteretes/cognee/blob/main/examples/python/ontology_demo_example.py) - Shows fundamental ontology integration
* [Advanced ontology demo](https://github.com/topoteretes/cognee/blob/main/examples/python/ontology_demo_example_2.py) - Demonstrates more complex ontology workflows


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt