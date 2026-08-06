import os

import modal

import dotenv

dotenv.load_dotenv()

# --- Configuration ---
CORPUS_FILE = "hotpot_qa_24_corpus.json"
QA_PAIRS_FILE = "hotpot_qa_24_qa_pairs.json"
INSTANCE_FILTER_FILE = "hotpot_qa_24_instance_filter.json"

# --- Shared Image Definition ---
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "cognee==0.2.0",
        "deepeval==3.2.6",
        "python-dotenv>=0.9.9",
        "gdown>=5.2.0",
        "langchain-openai>=0.3.28",
        "lightrag-hku[api]>=1.4.1",
        "mem0ai>=0.1.114",
        "nano-vectordb>=0.0.4.3",
        "openai>=1.97.0",
        "plotly>=6.2.0",
    )
    .env(
        {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "LLM_API_KEY": os.environ.get("LLM_API_KEY", ""),
            "LLM_MODEL": os.environ.get("LLM_MODEL", ""),
        }
    )
    .add_local_dir("qa", remote_path="/root/qa")
    .add_local_dir("modal_apps", remote_path="/root/modal_apps")
    .add_local_file(CORPUS_FILE, f"/root/{CORPUS_FILE}")
    .add_local_file(QA_PAIRS_FILE, f"/root/{QA_PAIRS_FILE}")
    .add_local_file(INSTANCE_FILTER_FILE, f"/root/{INSTANCE_FILTER_FILE}")
)

# --- Graphiti-specific Image Definition ---
graphiti_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "python-dotenv>=0.9.9",
        "graphiti-core==0.17.8",
        "langchain-openai>=0.3.28",
        "deepeval>=3.2.8",
        "plotly>=6.2.0",
        "openai>=1.97.0",
        "neo4j>=5.28.1",
    )
    .env(
        {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "LLM_API_KEY": os.environ.get("LLM_API_KEY", ""),
        }
    )
    .add_local_dir("qa", remote_path="/root/qa")
    .add_local_dir("modal_apps", remote_path="/root/modal_apps")
    .add_local_file(CORPUS_FILE, f"/root/{CORPUS_FILE}")
    .add_local_file(QA_PAIRS_FILE, f"/root/{QA_PAIRS_FILE}")
    .add_local_file(INSTANCE_FILTER_FILE, f"/root/{INSTANCE_FILTER_FILE}")
)

# --- Neo4j Image Definition ---
neo4j_env_dict = dict(
    NEO4J_AUTH="neo4j/pleaseletmein",
    NEO4J_ACCEPT_LICENSE_AGREEMENT="yes",
    NEO4J_PLUGINS='["apoc", "graph-data-science"]',
)

neo4j_image = (
    modal.Image.from_dockerfile("modal_apps/Dockerfile.neo4j-custom")
    .env(neo4j_env_dict)
    .add_local_dir("modal_apps", remote_path="/root/modal_apps")
)
