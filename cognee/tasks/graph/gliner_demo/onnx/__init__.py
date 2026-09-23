"""ONNX Runtime backend for the GLiNER demo extractor.

``export.py`` turns the three neural pieces of a loaded ``gliner2`` boundary
extractor into ONNX graphs (needs torch; run once per model). ``runtime.py``
swaps those pieces for ONNX Runtime sessions inside a loaded extractor, leaving
gliner2's own tokenization, batching and decoding untouched.
"""
