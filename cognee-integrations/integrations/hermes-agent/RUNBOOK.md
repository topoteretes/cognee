# Runbook: recovering from a degraded search index (Ollama embedding overflow)

**Symptoms.** Ingestion reports success and the dataset says processing
completed, but semantic retrieval returns nothing useful or times out, and the
server log shows repeated `Ollama embedding error: input length exceeds …`
lines:

```bash
grep -i "ollama embedding error" ~/.cognee-plugin/hermes/server.log
```

**Cause.** cognee sized its chunks from an `EMBEDDING_MAX_COMPLETION_TOKENS`
above the embedding model's real context length. Every oversized text overflowed
the model; cognee split it and mean-pooled the vectors while still reporting
success, so the vector index filled with lossy embeddings. Writes made after the
env is fixed are fine — but everything already written with the wrong settings
stays wrong until re-embedded, which is why the rebuild below is not optional.

## Recovery

1. **Pause ingestion.** Stop any backfill cron/batch job feeding the affected
   dataset before touching anything else — new writes during the rebuild would
   mix good and bad vectors again.

2. **Fix the embedding env** in `$HERMES_HOME/.env` (see
   [.env.example](./.env.example)). The ceiling must be at or below the model's
   context length (`ollama show <model>`), and the tokenizer must match the
   model:

   ```bash
   EMBEDDING_MAX_COMPLETION_TOKENS=256   # all-minilm; 2048 for nomic-embed-text
   HUGGINGFACE_TOKENIZER=sentence-transformers/all-MiniLM-L6-v2
   ```

   For the models the plugin recognizes, these pins are applied automatically at
   server spawn — this step is only needed for other models, or to override.

3. **Stop the running cognee server** so the new env applies at respawn — a
   server keeps the environment it was started with:

   ```bash
   curl -s http://127.0.0.1:8011/health   # confirm which port your server owns
   lsof -ti :8011 | xargs kill            # adjust the port if you changed COGNEE_LOCAL_PORT
   ```

   > The 8011 server is **shared with the other cognee plugins** (Claude Code,
   > Codex, OpenClaw). Stop it when their sessions are idle; they respawn it on
   > their next use.

4. **Rebuild the affected dataset.** Delete it and re-ingest the sources —
   re-running ingestion without deleting re-embeds nothing:

   - via the tool: `cognee_forget` with the dataset name, then re-`remember`
     each source record;
   - or over HTTP: `POST /api/v1/forget` with `{"dataset": "<name>"}`, then
     `POST /api/v1/remember` per record.

5. **Verify before resuming.** All three must hold:

   - no new `Ollama embedding error` lines appear in
     `~/.cognee-plugin/hermes/server.log` during re-ingestion;
   - a `cognee_recall` with `search_type=CHUNKS` for known content returns the
     stored text within seconds;
   - a default (`GRAPH_COMPLETION`) recall returns a cited answer within your
     `COGNEE_RECALL_TIMEOUT`.

6. **Resume ingestion** (restart the backfill job) only once step 5 passes.

## If recall is slow but the index is healthy

The default `GRAPH_COMPLETION` search runs an LLM per query — slow on local
models even with a good index. Use `search_type=CHUNKS` for fast raw-text
retrieval, or raise `COGNEE_RECALL_TIMEOUT`.
