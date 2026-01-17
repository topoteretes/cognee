# Ontology Quickstart

> Step-by-step guide to using OWL ontologies to ground Cognee knowledge graphs

A minimal guide to using OWL ontologies to ground Cognee's knowledge graphs. You'll point Cognee at an ontology file during cognify and then ask ontology-aware questions.

**Before you start:**

* Complete [Quickstart](getting-started/quickstart) to understand basic operations
* Read [Ontologies](../core-concepts/further-concepts/ontologies) to understand the concepts
* Ensure you have [LLM Providers](setup-configuration/llm-providers) configured
* Have an OWL ontology file (`.owl`) in RDF/XML format
* Have some text or files relevant to the ontology's domain

## What Ontology Support Does

* Grounds entities and relations to your OWL ontology (classes, individuals, properties)
* Validates types via ontology domains/ranges and class hierarchy
* Improves graph completion answers for domain-specific queries

## Step 1: Prepare an Ontology File

Start from a simple OWL file. Minimal ingredients:

* Classes (e.g., `TechnologyCompany`, `Car`)
* Individuals (e.g., `Apple`, `Audi`)
* Object properties with domain/range (e.g., `produces` with `domain=CarManufacturer`, `range=Car`)

Example ontology files:

* `examples/python/ontology_input_example/basic_ontology.owl`
* `examples/python/ontology_input_example/enriched_medical_ontology_with_classes.owl`

<Tip>
  Use any RDF/OWL editor (Protégé) to edit .owl files.
</Tip>

<Note>
  This example uses a simple ontology for demonstration. In practice, you can work with larger, more complex ontologies - the same approach works regardless of ontology size or complexity.
</Note>

## Step 2: Add Your Data

Add either raw text or a directory. Keep it relevant to your ontology.

```python  theme={null}
import cognee

texts = [
    "Audi produces the R8 and e-tron.",
    "Apple develops iPhone and MacBook."
]

await cognee.add(texts)
# or: await cognee.add("/path/to/folder/of/files")
```

<Note>
  This simple example uses a list of strings for demonstration. In practice, you can add multiple documents, files, or entire datasets - the ontology processing works the same way across all your data.
</Note>

## Step 3: Cognify Your Data + Ontologies

Create the `config` which contains the information about the ontology,
to ground extracted entities/relations to the ontology.
Then, simply pass the `config` to the `cognify` operation.

```python  theme={null}
import os
from cognee.modules.ontology.ontology_config import Config
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver

ontology_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ontology_input_example/basic_ontology.owl"
)

# Create full config structure manually
config: Config = {
    "ontology_config": {
        "ontology_resolver": RDFLibOntologyResolver(ontology_file=ontology_path)
    }
}

await cognee.cognify(config=config)
```

<Info>
  If omitted, Cognee builds a graph without ontology grounding. With an ontology, Cognee aligns nodes to classes/individuals and enforces property domain/range.
</Info>

## Step 4: Ask Ontology-aware Questions

Use `SearchType.GRAPH_COMPLETION` to get answers that leverage ontology structure.

```python  theme={null}
from cognee.api.v1.search import SearchType

result = await cognee.search(
    query_type=SearchType.GRAPH_COMPLETION,
    query_text="What cars and their types are produced by Audi?",
)
print(result)
```

<Tip>
  Phrase questions using ontology terms (class names, individual names, property language like "produces", "develops"). If results feel generic, check that the ontology contains the expected classes/individuals and that your data mentions them.
</Tip>

## Code in Action

* Small cars/tech demo: `examples/python/ontology_demo_example.py`
* Medical comparison demo: `examples/python/ontology_demo_example_2.py`

<Columns cols={3}>
  <Card title="Core Concepts" icon="brain" href="/core-concepts/further-concepts/ontologies">
    Understand ontology fundamentals
  </Card>

  <Card title="API Reference" icon="code" href="/api-reference/introduction">
    Explore ontology API endpoints
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt