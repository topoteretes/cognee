# Embedding quality eval

Compares any embedding model's retrieval with `text-embedding-3-large` on War and
Peace, for questions about weddings and marriages. It grew out of the BROAD search
work (SDK-324), where the default local model (FastEmbed `BAAI/bge-small-en-v1.5`)
never ranked the passage where Pierre marries Hélène above about 550 of 1,500 chunks,
while `text-embedding-3-large` brought it to about 60.

## How it works

1. The book (Project Gutenberg #2600) is cut once into passages of at most 510 tokens,
   the limit of the default local model, and saved to `passages.jsonl`.
2. `build` creates a persistent cognee environment for one embedding model. Every
   passage is added as its own document and stored by the indexing pipeline (chunk +
   embed, no LLM), so each passage is exactly one chunk and every environment holds the
   same texts. The build fails if a model's tokenizer splits a passage.
3. `compare` ranks every chunk for twelve wedding questions in both environments with
   cognee's vector search and reports:
   - where each of nine known wedding passages ranks, per question;
   - how many question-passage pairs land in the top `k` (default 30);
   - how many passages at least one question finds in the top `k`;
   - how much of the reference's top `k` the candidate also returns.

Ranks are over the whole book, so any `k` can be read off the report.

## Run

```bash
export LLM_API_KEY=...   # only for hosted models; never written to the env files
python cognee/tests/performance/embeddings/embedding_quality_eval.py --workdir ~/embedding-envs \
    build --name openai-3-large --provider openai \
    --model openai/text-embedding-3-large --dimensions 3072
python cognee/tests/performance/embeddings/embedding_quality_eval.py --workdir ~/embedding-envs \
    build --name fastembed-bge-small --provider fastembed \
    --model BAAI/bge-small-en-v1.5 --dimensions 384
python cognee/tests/performance/embeddings/embedding_quality_eval.py --workdir ~/embedding-envs \
    compare --reference openai-3-large --candidate fastembed-bge-small
```

Any provider cognee supports works as a candidate; pass extra settings with
`--setting KEY=VALUE` (for example `EMBEDDING_ENDPOINT` for Ollama). Each report is
also written as JSON to `<workdir>/reports/`.

Environments are kept. To use one for other experiments, point cognee at its settings
with `COGNEE_ENV_FILE=<workdir>/environments/<name>/.env` and search the
`war_and_peace` dataset.
