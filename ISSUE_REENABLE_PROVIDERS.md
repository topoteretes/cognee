Title: Finish live integration for google/azure/llm translation providers

Summary
-------
This PR temporarily unregisters the `google`, `azure`, and `llm` translation
providers to keep the current change small and reviewable. Those providers
require additional live integration testing, dependency pin resolution, and
secrets configuration before they should be merged. This issue tracks the work
needed to fully enable them.

Files of interest
-----------------
- `cognee/tasks/translation/translation_providers/google_provider.py`
- `cognee/tasks/translation/translation_providers/azure_provider.py`
- `cognee/tasks/translation/translation_providers/llm_provider.py`

Repro steps (local)
-------------------
1. From project root, activate your venv and install the following (adjust as
   necessary to avoid dependency conflicts):

   ```powershell
   C:/path/to/venv/Scripts/python.exe -m pip install googletrans==4.0.0rc1
   C:/path/to/venv/Scripts/python.exe -m pip install azure-ai-translation-text
   # LLM provider requires LLM API keys; no pip package required specifically.
   ```

2. Set required environment variables in a `.env` file or in your shell:
   - `LLM_API_KEY` for the LLM provider (provider-specific keys may be required)
   - `AZURE_TRANSLATE_KEY`, `AZURE_TRANSLATE_ENDPOINT`, `AZURE_TRANSLATE_REGION` (if required)

3. Run the probe script to verify provider instantiation:

   ```powershell
   $env:PYTHONPATH = 'C:\Users\DELL\Desktop\open\cognee'
   C:/path/to/venv/Scripts/python.exe scripts/list_translation_providers.py
   ```

4. If the Google provider fails with an httpcore/httpx compatibility error,
   experiment with compatible `httpx`/`httpcore` versions and re-run `pip
   install` until `googletrans` can import.

Acceptance criteria
-------------------
- Each provider instantiates without ImportError.
- For Google and Azure: detection and translation methods succeed when valid
  credentials are provided.
- For LLM: smoke test `scripts/smoke_gemini_test.py` runs successfully with a
  valid `LLM_API_KEY` (set as a GitHub secret for CI), and does not leak secrets
  in logs.

Notes
-----
- When re-enabling providers, prefer adding integration tests that run only in
  CI with secrets (via GitHub Actions secrets) and are gated behind a
  `RUN_LIVE_PROVIDER_TESTS` flag or similar to avoid accidental local runs.
- If the Google dependency resolution proves fragile, consider wrapping the
  google provider imports in a lightweight adapter that falls back to a
  stable REST-based translation API or external microservice.
