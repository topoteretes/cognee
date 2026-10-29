# Sourced at the start of a test step that also runs without repository
# secrets (fork PRs, and test_suites runs dispatched with simulate-fork):
#
#   source .github/scripts/keyless_ci_env.sh            # keyless: GLiNER + fastembed
#   source .github/scripts/keyless_ci_env.sh mock-llm   # test patches the LLM itself
#
# With secrets (CI_HAS_SECRETS=true, also the default when unset) it does
# nothing, so the step runs exactly as before.
#
# Without secrets, GitHub still sets every `${{ secrets.X }}` variable, to an
# empty string, and the workflows add literals such as EMBEDDING_DIMENSIONS and
# COGNEE_SKIP_CONNECTION_TEST. cognee counts an empty provider setting as
# configured, and the skip variables mean "honour my config", so any of them
# turns off the keyless local defaults. Removing them all leaves the
# configuration a new user gets with no key.
#
# `mock-llm` is for tests that patch the LLM call themselves: it then sets
# MOCK_EMBEDDING=true, which keeps cognee on the (patched) LLM extractor and
# replaces embeddings with deterministic vectors.

if [ "${CI_HAS_SECRETS:-true}" != "true" ]; then
  for _cognee_ci_var in $(compgen -e); do
    case "$_cognee_ci_var" in
      LLM_* | EMBEDDING_* | MOCK_EMBEDDING | COGNEE_SKIP_PREFLIGHT | COGNEE_SKIP_CONNECTION_TEST)
        unset "$_cognee_ci_var"
        ;;
    esac
  done
  unset _cognee_ci_var
  if [ "${1:-}" = "mock-llm" ]; then
    export MOCK_EMBEDDING=true
    echo "No repository secrets: provider settings cleared, LLM patched by the test, embeddings mocked."
  else
    echo "No repository secrets: provider settings cleared, cognee runs keyless (GLiNER + fastembed)."
  fi
fi
