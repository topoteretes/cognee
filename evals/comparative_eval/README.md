# Comparative QA Benchmarks

Independent benchmarks for different QA/RAG systems using HotpotQA dataset.

## Dataset Files
- `hotpot_50_corpus.json` - 50 instances from HotpotQA
- `hotpot_50_qa_pairs.json` - Corresponding question-answer pairs

## Benchmarks

Each benchmark can be run independently with appropriate dependencies:

### Mem0
```bash
pip install mem0ai openai
python qa_benchmark_mem0.py
```

### LightRAG
```bash
pip install "lightrag-hku[api]"
python qa_benchmark_lightrag.py
```

### Graphiti
```bash
pip install graphiti-core
python qa_benchmark_graphiti.py
```

## Environment
Create `.env` with required API keys:
- `OPENAI_API_KEY` (all benchmarks)
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (Graphiti only)

## Usage
Each benchmark inherits from `QABenchmarkRAG` base class and can be configured independently.

# Results
Updated results will be posted soon.
