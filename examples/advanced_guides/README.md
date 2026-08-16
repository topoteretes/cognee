# Cognee Advanced Guides

Deeper companions to the scripts in [`../guides/`](../guides/). Each one keeps
the same core idea as its guide counterpart but pushes it further: real bundled
documents instead of inline strings, longer multi-session threads, and — where
it makes sense — an explicit PASSED/FAILED verdict on the behaviour under test.

| Script | Advanced companion to | What it adds |
|---|---|---|
| [`remember_recall_improve_example.py`](remember_recall_improve_example.py) | [`guides/simple_cognee_example.py`](../guides/simple_cognee_example.py), [`guides/improve_quickstart.py`](../guides/improve_quickstart.py) | Tours the whole memory API surface (`remember`, `recall`, `improve`, `forget`, `status`) in nine steps |
| [`simple_document_qa/simple_document_qa_demo.py`](simple_document_qa/simple_document_qa_demo.py) | [`guides/simple_cognee_example.py`](../guides/simple_cognee_example.py) | Q&A over a bundled real document (the full text of Alice in Wonderland) instead of a short inline string |
| [`conversation_session_persistence_example.py`](conversation_session_persistence_example.py) | [`guides/sessions.py`](../guides/sessions.py) | Six turns across two sessions, persisted into the permanent knowledge graph, then rendered |
| [`session_distillation_demo.py`](session_distillation_demo.py) | [`guides/session_distillation.py`](../guides/session_distillation.py) | Replays an eight-message scripted session with hybrid vector recall and verifies the distilled lessons landed in the graph |
| [`temporal_awareness_example/temporal_awareness_example.py`](temporal_awareness_example/temporal_awareness_example.py) | [`guides/temporal_recall.py`](../guides/temporal_recall.py) | Temporal search over two bundled real biographies: before / after / between ranges plus person-centric timelines |
| [`ontology_reference_vocabulary/ontology_as_reference_vocabulary_example.py`](ontology_reference_vocabulary/ontology_as_reference_vocabulary_example.py) | [`guides/ontology_quickstart.py`](../guides/ontology_quickstart.py) | Grounds two bundled real documents in an OWL ontology, building the ontology `Config` with `RDFLibOntologyResolver` explicitly |
| [`global_context_index_smoke_demo.py`](global_context_index_smoke_demo.py) | [`guides/global_context_index.py`](../guides/global_context_index.py) | Multi-day scheduling thread whose answers depend on the whole history; reports a PASSED/FAILED verdict on the context prelude |
| [`truth_centroid_slots_demo.py`](truth_centroid_slots_demo.py) | [`../demos/truth_subspace_reranking_demo.py`](../demos/truth_subspace_reranking_demo.py) | How truth-subspace re-ranking works underneath: deterministic centroid slots, epochs, and rebuilds across learning batches |

## Running

```bash
uv run python examples/advanced_guides/<script>.py
```

Requires `LLM_API_KEY` in `.env` (copy `.env.template`). The sub-directory
examples load their data from the bundled `data/` folder next to the script —
no external files needed.
