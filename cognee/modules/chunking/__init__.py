"""Text chunkers and chunk identity.

``TextChunker`` (default), ``LangchainChunker``, ``CsvChunker`` and
``JsonListChunker`` implement ``Chunker``; ``chunk_id.py`` derives the
content-based chunk id (``uuid5(doc : sha256(text) : occurrence)``) that lets
unchanged content keep its identity across edits; ``incremental_chunking.py``
is the diff + re-chunk used by ``update()``; ``models/`` holds
``DocumentChunk``. Chunk-level tasks live in ``cognee.tasks.chunks``.
"""
