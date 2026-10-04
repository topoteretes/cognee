#!/bin/sh
# Refuse to start on a backend URL the UI cannot use.
#
# A container that cannot resolve its backend should fail visibly at startup,
# with the reason in `docker logs`, rather than boot and answer every request
# with a 500 that says nothing about the cause. Under a restart policy this
# turns a silent misconfiguration into an obvious crash loop.
#
# Checked here rather than inside the app because it must run before the server
# does, and because the Next build rejects process.exit in its instrumentation
# hook. The rules below are the same ones normalizeBackendUrl applies in
# src/modules/config/runtimeConfig.ts, expressed in node rather than in shell
# globs so that trimming, the case-insensitive scheme and a bare "http://" all
# behave identically in both places. Keep the two in step.
set -e

node -e '
  const raw = process.env.COGNEE_BACKEND_URL;
  const value = (raw ?? "").trim();
  if (!value) process.exit(0);

  const fail = (message) => {
    console.error("[cognee-ui] COGNEE_BACKEND_URL " + message);
    process.exit(1);
  };

  if (!/^https?:\/\//i.test(value)) {
    fail(`must be an absolute http(s) URL, got "${value}". Expected something like "http://localhost:8000".`);
  }
  try {
    new URL(value);
  } catch {
    fail(`is not a valid URL: "${value}".`);
  }
'

exec "$@"
