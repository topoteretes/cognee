"""
Document classification and chunking tasks.

This module provides tasks for classifying raw data into document objects
based on file extensions and extracting structured chunks from documents
for downstream processing and storage.
"""

from .classify_documents import classify_documents
from .extract_chunks_from_documents import extract_chunks_from_documents
