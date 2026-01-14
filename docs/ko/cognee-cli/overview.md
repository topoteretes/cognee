# Cognee CLI Overview

> Command line interface for Cognee AI memory operations

The `cognee-cli` command lets you run Cognee from the terminal so you can add data, build the knowledge graph, and ask questions without opening a Python file. The commands are designed to be short, use friendly defaults, and are safe for people who are just starting out.

## Setup

Before using the CLI, you need to configure your API key. The recommended approach is to store it in a `.env` file:

```bash  theme={null}
# Create a .env file in your project root
echo "LLM_API_KEY=your_openai_api_key" > .env
```

Alternatively, you can export it in your terminal session:

```bash  theme={null}
export LLM_API_KEY=your_openai_api_key
```

<Note>
  Use the `cognee-cli config set` command only for temporary tweaks during a long-running session. For persistent configuration, use `.env` files or environment variables.
</Note>

## Quick Tour of Commands

* `cognee-cli add <data>` loads documents or text into a dataset
* `cognee-cli cognify` turns datasets into a knowledge graph
* `cognee-cli search "question"` asks the graph for answers
* `cognee-cli delete` removes stored data when you no longer need it
* `cognee-cli config` reads and updates saved settings
* `cognee-cli -ui` launches the local web app

Add `--help` after any command (for example, `cognee-cli search --help`) to see every option.

## Add Data

Start by loading something the graph can learn from. You can add files, folders, URLs, or even plain text.

```bash  theme={null}
# Add a single file to the default dataset
cognee-cli add docs/company-handbook.pdf

# Pick a dataset name so you can separate topics later
cognee-cli add docs/policies.docx --dataset-name onboarding

# Add multiple files at once
cognee-cli add docs/policies.docx docs/faq.md --dataset-name onboarding

# Add a short text note (wrap the note in quotes)
cognee-cli add "Kickoff call notes: customer wants faster onboarding" --dataset-name sales_calls
```

<Accordion title="Add Command Options">
  * `data`: One or more file paths, URLs, or text strings. Mix and match as needed
  * `--dataset-name` (`-d`): Defaults to `main_dataset`. Use clear names so the team remembers what each dataset holds
</Accordion>

## Cognify Data

Cognify builds the knowledge graph. Run it whenever you add new data or change the ontology.

```bash  theme={null}
# Process every dataset
cognee-cli cognify

# Process specific datasets only
cognee-cli cognify --datasets onboarding sales_calls

# Increase chunk size and show more logs
cognee-cli cognify --datasets onboarding --chunk-size 1500 --chunker TextChunker --verbose

# Kick off a long job and return immediately
cognee-cli cognify --datasets onboarding --background
```

<Accordion title="Cognify Command Options">
  * `--datasets` (`-d`): Space-separated list. Skip it to process everything
  * `--chunk-size`: Token limit for each chunk. Leave blank to let Cognee choose
  * `--chunker`: `TextChunker` (default) or `LangchainChunker` if installed
  * `--background` (`-b`): Handy for large datasets; the CLI exits while the job keeps running
  * `--verbose` (`-v`): Prints progress messages
  * `--ontology-file`: Path to a custom ontology (`.owl`, `.rdf`, etc.)
</Accordion>

## Search the Graph

Once cognify finishes, you can question the graph. Start with a simple natural-language question, then experiment with search types.

```bash  theme={null}
# Default search (GRAPH_COMPLETION)
cognee-cli search "Who owns the rollout plan?"

# Limit the scope to one dataset
cognee-cli search "What is the onboarding timeline?" --datasets onboarding

# Return three answers at most
cognee-cli search "List the key risks" --top-k 3

# Save a JSON response for another tool
cognee-cli search "Which documents mention security?" --output-format json
```

<Accordion title="Search Types">
  Try these quick examples to feel the differences:

  ```bash  theme={null}
  # Conversational answer with reasoning (default)
  cognee-cli search "Give me a summary of onboarding" --query-type GRAPH_COMPLETION

  # Shorter answer based on chunks
  cognee-cli search "Show the onboarding steps" --query-type RAG_COMPLETION

  # Highlight relationships and insights
  cognee-cli search "How do onboarding tasks connect?" --query-type INSIGHTS

  # Raw text passages you can copy
  cognee-cli search "Find security requirements" --query-type CHUNKS --top-k 5

  # Summaries only (great for reviews)
  cognee-cli search "Summarise the onboarding handbooks" --query-type SUMMARIES

  # Code-aware search for repos
  cognee-cli search "Where is the email parser?" --query-type CODE

  # Advanced graph query (requires Cypher skills)
  cognee-cli search "MATCH (n) RETURN COUNT(n)" --query-type CYPHER
  ```
</Accordion>

<Accordion title="Search Command Options">
  * `--query-type`: Choose from GRAPH\_COMPLETION, RAG\_COMPLETION, INSIGHTS, CHUNKS, SUMMARIES, CODE, or CYPHER
  * `--datasets`: Limit search to specific datasets
  * `--top-k`: Maximum number of results to return
  * `--system-prompt`: Point to a custom prompt file for LLM-backed modes
  * `--output-format` (`-f`): `pretty` (friendly layout), `simple` (minimal text), or `json` (structured output for scripts)
</Accordion>

## Delete Data

Clean up when a dataset is outdated or when you reset the environment.

```bash  theme={null}
# Remove one dataset (asks for confirmation)
cognee-cli delete --dataset-name onboarding

# Remove everything for a specific user
cognee-cli delete --user-id 123e4567

# Wipe all data (add --force to skip the question)
cognee-cli delete --all --force
```

<Accordion title="Delete Command Options">
  * `--dataset-name`: Remove a specific dataset
  * `--user-id`: Remove all data for a specific user
  * `--all`: Remove all data (use with caution)
  * `--force`: Skip confirmation prompts
</Accordion>

## Manage Configuration

The CLI stores its settings so you do not have to repeat them. Configuration updates line up with the Python API.

```bash  theme={null}
# See the list of supported keys
cognee-cli config list

# Check one value (if implemented)
cognee-cli config get llm_model

# Update your LLM provider and model
cognee-cli config set llm_provider openai
cognee-cli config set llm_model gpt-4o-mini

# Store an API key (quotes are optional)
cognee-cli config set llm_api_key sk-yourkey

# Reset a key back to its default value
cognee-cli config unset chunk_size
```

<Accordion title="Config Command Options">
  * `list`: Print the common keys
  * `get [key]`: Show the saved value; omit the key to list everything
  * `set <key> <value>`: Save a new value. JSON strings such as `{}` or `true` are parsed automatically
  * `unset <key>`: Reset to the default. Add `--force` to skip confirmation
  * `reset`: Placeholder for a future "reset everything" command
</Accordion>

<Accordion title="Useful Configuration Keys">
  * Language model: `llm_provider`, `llm_model`, `llm_api_key`, `llm_endpoint`
  * Storage: `graph_database_provider`, `vector_db_provider`, `vector_db_url`, `vector_db_key`
  * Chunking: `chunk_size`, `chunk_overlap`
</Accordion>

## Launch the UI

Prefer a browser view? Launch the UI with one flag.

```bash  theme={null}
cognee-cli -ui
```

The CLI starts the backend on `http://localhost:8000` and the React app on `http://localhost:3000`. Leave the window open and press `Ctrl+C` to stop everything.

## Next Steps

<CardGroup cols={2}>
  <Card title="Installation Guide" href="/getting-started/installation" icon="download">
    **Set up your environment**

    Install Cognee and configure your environment to start using the CLI.
  </Card>

  <Card title="Quickstart Tutorial" href="/getting-started/quickstart" icon="play">
    **Run your first example**

    Get started with Cognee by running your first knowledge graph example.
  </Card>
</CardGroup>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt