# POC Disambiguation

This folder contains proof-of-concept scripts and assets focused on entity disambiguation during
knowledge graph extraction. The POC injects existing entity names into the LLM prompt to bias
extraction toward reusing canonical entities instead of creating duplicates.

## What Was Done

- Added a custom graph extraction step that performs a vector lookup over existing entity names
  and appends those candidates to the extraction prompt.
- Wired that custom extraction into a full `cognify`-style pipeline to compare against the
  default behavior.
- Provided small synthetic datasets that include aliasing, abbreviations, and spelling variants
  to make the disambiguation effect easy to observe.
- Included a runnable example that builds and visualizes graphs with and without the POC.

## Scripts And What They Do

`disambiguate_entities.py`
- Provides `disambiguate_entities_pipeline(...)`, a pipeline equivalent to `cognee.cognify()` but
  with the graph extraction task swapped for the POC entity-disambiguation extractor.

Parameters (differences vs `cognee.cognify()`):
- `custom_prompt`: Optional prompt string appended with candidates from vector search.
- `serach_limit`: Typo in signature; currently unused.
- `temporal_cognify`: Present but unused in this POC.
- `**kwargs`: Passed through to the POC extractor (used for `vector_search_limit`).

`poc_extract_graph_from_data_with_entity_disambiguation.py`
- Implements `extract_graph_from_data_with_entity_disambiguation(...)` which:
  - Looks up existing entities from the vector DB collection `Entity_name`.
  - Appends the top matches to the prompt (`custom_prompt`) as candidate aliases.
  - Runs `extract_content_graph(...)` for each chunk using the updated prompt.
  - Filters invalid edges when using the default `KnowledgeGraph` model.
  - Integrates chunk graphs and adds nodes/edges via `add_data_points`.

Parameters (differences vs `extract_graph_from_data`):
- No `context` parameter.
- `custom_prompt`: Prompt string to seed the LLM; will be extended with vector matches.
- `vector_search_limit` (via `**kwargs`): Number of existing entities to retrieve (default 5).

`poc_disambiguate_entities_example.py`
- End-to-end runnable example that:
  - Prunes existing Cognee data.
  - Reads a small example file line-by-line from `poc_disambiguation/data/`.
  - Adds each line as a separate datapoint.
  - Runs either the POC pipeline (`use_poc=True`) or the standard `cognee.cognify()`.
  - Writes graph visualizations to `poc_disambiguation/results/`.

Key parameters:
- `use_poc`: If true, runs `disambiguate_entities_pipeline` after each added line.
- `vector_search_limit`: Number of candidate entities appended to the prompt.
- `custom_prompt`: Prompt template read from `prompts/prompt1.txt`.

Outputs:
- `results/cognify_simple_<example>_poc_graph.html`
- `results/cognify_simple_<example>_graph.html`

## Examples And Data

`data/example1.txt`
- Variants of “OpenAI” (spacing, punctuation, paraphrases).

`data/example2.txt`
- “NASA” variants including abbreviations and formal name.

`data/example3.txt`
- Person name spelling/spacing variants.

`data/example4.txt`
- Location name variants (NYC, New York, etc.).

`prompts/prompt1.txt`
- Base prompt for graph extraction with explicit alias-reuse rules. The POC appends existing
  entity names under the “Existing entities:” section.
