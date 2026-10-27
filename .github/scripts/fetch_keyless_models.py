"""Load the local models a keyless cognify uses (GLiNER, fastembed), filling their caches."""

import asyncio

from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine
from cognee.tasks.graph.gliner_demo.extractor import load_extractor

load_extractor()
asyncio.run(get_embedding_engine().embed_text(["warm-up"]))
