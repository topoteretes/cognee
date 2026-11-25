// Open-source override — the SaaS barrel also re-exports the pod and
// management HTTP clients, which are cloud-only and excluded from the sync.
// Everything else it exports has to stay listed here: this file replaces the
// barrel wholesale, so an export dropped from it is not inherited from the
// SaaS one, it is gone, and the modules importing it fail at build time with
// no earlier signal (verify-public-imports.js resolves module paths, not the
// names inside them).
//
// SLOW_READ_TIMEOUT_MS: ./client.ts does not define this on this branch yet.
// It lands with the dev merge, together with the two @/modules/business
// consumers that need it, so it belongs here from the same commit that pulls
// those in -- listed now so the two land as one change instead of two. Until
// then this line does not resolve, but nothing in CI compiles this tree
// against real dev sources to notice: tsconfig excludes scripts/, eslint
// ignores it, and --verify-only checks import paths, not the names inside
// them. Do not merge this file's current state without that merge alongside it.
export { http, SLOW_READ_TIMEOUT_MS } from "./client";
export { HttpError, toHttpError, normalizeError } from "./errors";
export { reportClientLog } from "./reportClientLog";
