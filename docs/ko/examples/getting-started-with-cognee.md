# Cognee Walkthrough

> From Data to Interactive Memory: End-to-end tutorial with nodesets, ontologies, memify, graph visualization, and feedback system using a coding assistant example

Cognee gives you the tools to **build smarter AI agents** with context-aware memory.

Use it to create a **queryable knowledge graph** powered by embeddings from your data. When retrieving data, your agent can reach up to **92.5% accuracy**.

## What You'll Learn

In this tutorial, you'll:

* **Organize memory** with [nodesets](/core-concepts/further-concepts/node-sets) and apply filters during retrieval
* **Define your data model** using [ontology support](/guides/ontology-support)
* **Enhance memory** with contextual enrichment layers
* **Visualize your graph** with [graph visualization](/guides/graph-visualization) to explore stored knowledge
* **Search smarter** by combining vector similarity with graph traversal
* **Refine results** through interactive search and [feedback](/guides/feedback-system)

## Example Use Case

In this example, you will use a **Cognee-powered [Coding Assistant](/examples/code-assistants)** to get context-aware coding help.

You can open [this example on a Google Colab Notebook](https://colab.research.google.com/drive/12Vi9zID-M3fpKpKiaqDBvkk98ElkRPWy?usp=sharing) and run the steps shown below to build your cognee memory interactively.

## Prerequisites

* OpenAI API key (or another supported LLM provider)

> Cognee uses OpenAI's GPT-5 model as default. Note that the OpenAI free tier does not satisfy the rate limit requirements. Please refer to our [LLM providers documentation](https://docs.cognee.ai/setup-configuration/llm-providers) to use another provider.

## Setup

First, let's set up the environment and import necessary modules.

<Accordion title="Utility Functions Setup">
  Create a utility class to handle file downloads and visualization helpers:

  ```python  theme={null}
  class NotebookUtils:
      """Utility class for cognee demo - helper methods to keep the main notebook clean and focused."""
      
      def __init__(self):
          """Initialize the NotebookUtils with default configurations."""
          self.artifacts_dir = None
          self.assets_config = self._initialize_assets_config()
      
      def _initialize_assets_config(self) -> Dict[str, Tuple[str, str]]:
          """Initialize configuration mapping for remote assets to download from cognee repository."""
          return {
              "human_agent_conversations": (
                  "/content/copilot_conversations.json",
                  "https://raw.githubusercontent.com/topoteretes/cognee/main/notebooks/data/copilot_conversations.json",
              ),
              "python_zen_principles": (
                  "/content/zen_principles.md",
                  "https://raw.githubusercontent.com/topoteretes/cognee/main/notebooks/data/zen_principles.md",
              ),
              "ontology": (
                  "/content/basic_ontology.owl",
                  "https://raw.githubusercontent.com/topoteretes/cognee/main/examples/python/ontology_input_example/basic_ontology.owl",
              ),
          }
      
      def download_remote_file_if_not_exists(self, local_path: str, remote_url: str) -> str:
          """Download remote file if it doesn't exist locally to avoid unnecessary re-downloads."""
          file_path = Path(local_path)
          if not file_path.exists():
              file_path.parent.mkdir(parents=True, exist_ok=True)
              urllib.request.urlretrieve(remote_url, file_path)
              print(f"Downloaded: {file_path.name}")
          else:
              print(f"File already exists: {file_path.name}")
          return str(file_path)
      
      def load_json_file_content(self, file_path: str) -> Dict[str, Any]:
          """Load and parse JSON file content into a Python dictionary."""
          with open(file_path, "r", encoding="utf-8") as file:
              return json.load(file)
      
      def load_text_file_content(self, file_path: str) -> str:
          """Load and return raw text content from a text file."""
          with open(file_path, "r", encoding="utf-8") as file:
              return file.read()
      
      def preview_json_structure(self, json_data: Dict[str, Any], max_keys: int = 3) -> None:
          """Display formatted preview of JSON data structure and sample content."""
          print("JSON Structure Preview:")
          pprint.pp(list(json_data.keys())[:max_keys])
          if "conversations" in json_data and json_data["conversations"]:
              print("Sample conversation:")
              pprint.pp(json_data["conversations"][0])
      
      def preview_text_content(self, text_content: str, max_chars: int = 200) -> None:
          """Display formatted preview of text content to show its format."""
          print("Text Content Preview:")
          print(text_content[:max_chars])
          if len(text_content) > max_chars:
              print(f"... (truncated, total length: {len(text_content)} characters)")
      
      def create_notebook_artifacts_directory(self, dir_name: str = "artifacts") -> Path:
          """Create and return artifacts directory for storing notebook outputs like graph visualizations."""
          notebook_dir = Path.cwd()
          self.artifacts_dir = notebook_dir / dir_name
          self.artifacts_dir.mkdir(exist_ok=True)
          print(f"Artifacts directory created/verified at: {self.artifacts_dir}")
          return self.artifacts_dir
      
      def download_remote_assets(self) -> Dict[str, str]:
          """Download all remote assets from cognee repository and return their local file paths."""
          downloaded_assets = {}
          
          print("Downloading remote assets...")
          print("-" * 40)
          
          for asset_name, (local_path, remote_url) in self.assets_config.items():
              downloaded_assets[asset_name] = self.download_remote_file_if_not_exists(
                  local_path, remote_url
              )
          
          print("-" * 40)
          print(f"Successfully processed {len(downloaded_assets)} assets")
          return downloaded_assets
      
      def preview_downloaded_assets(self, asset_paths: Dict[str, str]) -> None:
          """Display comprehensive preview of all downloaded assets."""
          print("=== ASSET PREVIEWS ===\n")
          
          # Preview JSON files
          for asset_name, file_path in asset_paths.items():
              if file_path.endswith('.json'):
                  print(f"--- {asset_name.upper()} ---")
                  json_data = self.load_json_file_content(file_path)
                  self.preview_json_structure(json_data)
                  print()
          
          # Preview text files
          for asset_name, file_path in asset_paths.items():
              if file_path.endswith(('.md', '.txt')):
                  print(f"--- {asset_name.upper()} ---")
                  text_content = self.load_text_file_content(file_path)
                  self.preview_text_content(text_content)
                  print()
          
          # Preview OWL files
          for asset_name, file_path in asset_paths.items():
              if file_path.endswith('.owl'):
                  print(f"--- {asset_name.upper()} ---")
                  print(f"OWL ontology file: {Path(file_path).name}")
                  text_content = self.load_text_file_content(file_path)
                  self.preview_text_content(text_content, max_chars=300)
                  print()

  # Initialize the utility class
  utils = NotebookUtils()
  ```
</Accordion>

Install Cognee using pip:

```bash  theme={null}
!pip install cognee==0.3.4

# Create artifacts directory for storing visualization outputs
artifacts_path = utils.create_notebook_artifacts_directory()

import cognee
```

## Create Sample Data to Ingest into Memory

In this example, we'll use a **Python developer** scenario. The data sources we'll ingest into Cognee include:

* A short introduction about the developer (`developer_intro`)
* A conversation between the developer and a coding agent (`human_agent_conversations`)
* The Zen of Python principles (`python_zen_principles`)
* A basic ontology file with structured data about common technologies (`ontology`)

### Prepare the Sample Data

```python  theme={null}
# Define the developer introduction to simulate personal context
developer_intro = (
    "Hi, I'm an AI/Backend engineer. "
    "I build FastAPI services with Pydantic, heavy asyncio/aiohttp pipelines, "
    "and production testing via pytest-asyncio. "
    "I've shipped low-latency APIs on AWS, Azure, and GoogleCloud."
)

# Download additional datasets from the Cognee repository
asset_paths = utils.download_remote_assets()
human_agent_conversations = asset_paths["human_agent_conversations"]
python_zen_principles = asset_paths["python_zen_principles"]
ontology_path = asset_paths["ontology"]
```

The `download_remote_assets()` function:

* Handles multiple file types (JSON, Markdown, ontology)
* Creates the required folders automatically
* Prevents redundant downloads

## Review the Structure and Content of Downloaded Data

Next, let’s inspect the data we just downloaded.\
Use `preview_downloaded_assets()` to quickly summarize and preview each file’s structure and contents before Cognee processes them.

```python  theme={null}
# Preview each file's structure and contents
utils.preview_downloaded_assets(asset_paths)
```

## Reset Memory and Add Structured Data

Start by resetting Cognee's memory using `prune()` to ensure a clean, reproducible run.\
Then, use [`add()`](/core-concepts/main-operations/add) to load your data into dedicated node sets for organized memory management.

```python  theme={null}
await cognee.prune.prune_data()
await cognee.prune.prune_system(metadata=True)

await cognee.add(developer_intro, node_set=["developer_data"])
await cognee.add(human_agent_conversations, node_set=["developer_data"])
await cognee.add(python_zen_principles, node_set=["principles_data"])
```

## Configure the Ontology and Build a Knowledge Graph

Set the ontology file path, then run [`cognify()`](/core-concepts/main-operations/cognify) to transform all data into a **knowledge graph** backed by embeddings.\
Cognee automatically loads the ontology configuration from the `ONTOLOGY_FILE_PATH` environment variable.

```python  theme={null}
# Configure ontology file path for structured data processing
os.environ["ONTOLOGY_FILE_PATH"] = ontology_path

# Transform all data into a knowledge graph backed by embeddings
await cognee.cognify()
```

## Visualize and Inspect the Graph Before and After Enrichment

Generate HTML visualizations of your knowledge graph to see how Cognee processed the data.

First, visualize the initial graph structure. Then, use [`memify()`](/core-concepts/main-operations/memify) to enhance the knowledge graph adding deeper semantic connections and improves relationships between concepts. Finally, generate a second visualization to compare the enriched graph.

```python  theme={null}
# Generate initial graph visualization showing nodesets and ontology structure
initial_graph_visualization_path = str(artifacts_path / "graph_visualization_nodesets_and_ontology.html")
await cognee.visualize_graph(initial_graph_visualization_path)

# Enhance the knowledge graph with memory consolidation for improved connections
await cognee.memify()

# Generate second graph visualization after memory enhancement
enhanced_graph_visualization_path = str(artifacts_path / "graph_visualization_after_memify.html")
await cognee.visualize_graph(enhanced_graph_visualization_path)
```

The generated HTML files can be opened in your browser to explore and inspect the graph structure.

## Query Cognee Memory with Natural Language

Run cross-document [searches](/core-concepts/main-operations/search) to connect information across multiple data sources.\
Then, perform filtered searches within specific node sets to focus on targeted context.

```python  theme={null}
# Cross-document knowledge retrieval from multiple data sources
results = await cognee.search(
    query_text="How does my AsyncWebScraper implementation align with Python's design principles?",
    query_type=cognee.SearchType.GRAPH_COMPLETION,
)
print("Python Pattern Analysis:", results)

# Filtered search using NodeSet to query only specific subsets of memory
from cognee.modules.engine.models.node_set import NodeSet

results = await cognee.search(
    query_text="How should variables be named?",
    query_type=cognee.SearchType.GRAPH_COMPLETION,
    node_type=NodeSet,
    node_name=["principles_data"],
)
print("Filtered search result:", results)
```

## Provide Interactive Feedback for Continuous Learning

Run a search with `save_interaction=True` to capture user feedback.\
Then, use the `FEEDBACK` query type to refine future retrievals and improve Cognee’s performance over time.

```python  theme={null}
# Interactive search with feedback mechanism for continuous improvement
answer = await cognee.search(
    query_type=cognee.SearchType.GRAPH_COMPLETION,
    query_text="What is the most zen thing about Python?",
    save_interaction=True,
)
print("Initial answer:", answer)

# Provide feedback on the previous search result
# The last_k parameter specifies which previous answer to give feedback about
feedback = await cognee.search(
    query_type=cognee.SearchType.FEEDBACK,
    query_text="Last result was useful, I like code that complies with best practices.",
    last_k=1,
)
```

## Visualize the Graph After Feedback

Generate a final visualization to see how the feedback mechanism improved the knowledge graph.

```python  theme={null}
feedback_enhanced_graph_visualization_path = str(
    artifacts_path / "graph_visualization_after_feedback.html"
)

await cognee.visualize_graph(feedback_enhanced_graph_visualization_path)
```

This view highlights the enhanced connections and learning captured from user feedback.

## Next Steps

<CardGroup cols={2}>
  <Card title="Join the Community" href="https://discord.gg/cqF6RhDYWz" icon="discord">
    **Cognee Discord**

    Join over 1,000 builders to ask questions and share insights.
  </Card>

  <Card title="Explore Examples" href="https://github.com/topoteretes/cognee" icon="github">
    **GitHub Repository**

    Star our repo ⭐ and try additional examples to deepen your knowledge.
  </Card>
</CardGroup>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt