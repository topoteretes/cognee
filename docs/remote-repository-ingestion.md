# Remote repository ingestion

Remote repositories use a shallow, blobless partial clone. Cognee reads the complete
Git tree at a resolved commit, selects paths, and performs an exact sparse checkout.
A large repository can therefore fit on a small worker when most of its size comes
from excluded artifacts. There is no repository-size rejection threshold.

`remember(url, content_type="code")` excludes known binary artifacts and binary
document assets. Text, manifests, tracked hidden configuration and unknown extensions
remain available to Enola, including custom extractors. `add(url)` also checks out
assets supported by registered document loaders, such as images and PDFs. Its
document walker still applies its existing hidden-path rules. Both modes exclude
generated/dependency directories from `code_repo.SKIP_DIRS`, Git metadata, Enola
output, audio/video assets, symlinks and submodule contents.

The internal resolver's `include_documents` option defaults to `True`. Local paths
are returned unchanged and are not subject to managed-clone disk monitoring.

## Disk budget

Set either or both positive integer environment variables before ingestion:

| Setting | Meaning |
| --- | --- |
| `COGNEE_REPOS_MAX_BYTES` | Maximum measured bytes across the shared clones directory |
| `COGNEE_REPOS_MIN_FREE_BYTES` | Free-space reserve on the filesystem containing that directory |

Neither setting has an implicit default. Operators choose values from their worker's
actual storage allocation. Usage includes `.git`, checked-out files, other cached
repositories, temporary files and Enola output. Git and Enola scratch files are
placed on the measured volume. Admission, execution and completion are checked.
Exhaustion raises `RepositoryResourceError` (HTTP 507) with `reason` and `limit_bytes`,
and either `used_bytes` (budget exceeded) or `free_bytes` (free space too low); the
other is `None`. A value that is not a positive integer number of bytes, such as `5GB`,
raises a `CogneeConfigurationError` that names both settings. The measured clones
directory is rescanned at a pace of about a tenth of one worker thread, however large.
Running subprocesses and helpers are stopped and reaped before returning the error.
Failed new clones and resource-interrupted refreshes are removed; Enola output from
a resource-interrupted generation is removed too.

This is a polling guard, **not a hard filesystem quota**. Downloads can overshoot
between measurements. Cloud must set the budget below the pod's storage limit,
reserve space for other activity and enforce a storage quota independently. This SDK
change does not configure pods or implement billing/token admission. Deterministic
code ingestion makes no LLM calls unless additional embedding/enrichment is enabled;
document ingestion and later processing retain their existing costs.

## Explicit boundaries

- The server must advertise protocol-v2 `filter` support. Cognee refuses unsupported
  servers before cloning and interrupts a transfer if Git reports that the filter
  was ignored. It never retries as a full clone.
- Git LFS smudging is disabled. A selected LFS pointer raises an error; provide a
  local checkout with its content already materialized to ingest it. Excluded LFS
  assets are not downloaded.
- Submodules and symlinks are excluded and counted; submodules are never cloned
  recursively. A malformed or incomplete tree listing fails before checkout.
- Newlines in a selected filename are rejected because sparse patterns cannot
  represent them safely. Spaces and glob characters are escaped literally.
- Older full-clone caches may still contain previously fetched blobs. Sparse
  checkout does not reclaim their object packs. Remove an unused cache before
  retrying if those old packs consume its budget; do not remove an active clone.
- Snapshot metadata records `commit`, `include_documents`, `selected_files` and
  per-reason `skipped` counts in `.git/cognee-snapshot.json`, the repository manifest
  metadata and the code `remember()` result. This identifies checkout selection;
  it does not hold a lock through subsequent graph ingestion or provide resumability.
- One clone serves every caller of a repository URL. A later call with the other
  `include_documents` mode, or a refresh interrupted by the budget, adds or removes
  files in that clone, or removes it, while an earlier call may still be reading it.
  Route work on one repository URL to one worker, in one mode at a time.

See [Git partial clone options](https://git-scm.com/docs/git-clone),
[sparse checkout](https://git-scm.com/docs/git-sparse-checkout) and
[tree listing](https://git-scm.com/docs/git-ls-tree) for the underlying Git operations.
