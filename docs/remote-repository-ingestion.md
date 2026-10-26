# Remote repository ingestion

Use `add(repo_url, codegraph_config=...)`, then `cognify()` scoped to the data IDs
returned by add. See `examples/guides/remote_code_repository.py` for code-only
processing without an LLM key. Tokens belong in `repo_credentials`, not persisted URLs.

## Download and selection

Remote clones use shallow history, `--filter=blob:none` and an exact sparse
checkout at one commit. The tree is inspected before requesting selected blobs.
Symlinks, submodules, build/cache directories, audio/video and known binary
artifacts are excluded. Registered binary documents are included when
`include_documents=True`. Code-only mode retains text and tracked configuration,
including unknown extensions used by custom extractors. There is no file-size cutoff.
The ordinary document ingestion rules still apply after checkout.

Servers that ignore the filter fail rather than downloading a full clone.
Git LFS smudging is disabled; a selected LFS pointer raises an error asking for a
prepared local checkout. Paths containing newlines cannot be represented safely
by sparse checkout and raise an error.

The manifest records the commit, selection fingerprint, policy version, selected
file count and skip counts. Cognify holds the clone lock and validates both the
commit and selection. If another add changes either, re-add the repository to
that dataset before cognify. A manifest written before partial clones records no
selection and is refused the same way until it is re-added. Documents are copied
to durable storage during add.

An existing full managed clone is replaced under the same lock after a new
partial clone validates. Both coexist during preparation, so reserve space for
that transition. A failure while preparing the replacement preserves the old cache.
Once retirement starts, cancellation waits for the validated replacement to be
published before propagating; the clone lock stays held throughout. No unrelated
cache or existing dataset graph is deleted.

## Operator disk settings

Both settings are optional positive integers in bytes, read at operation time:

| Setting | What it measures |
| --- | --- |
| `COGNEE_REPOS_MAX_BYTES` | Combined managed clone storage, temporary clone/output files and local ingestion storage, counting nested roots once |
| `COGNEE_REPOS_MIN_FREE_BYTES` | Minimum free bytes on each destination filesystem |

Unset means no SDK disk limit. `COGNEE_REPOS_DIR` selects the clone root. For a
managed clone, Enola scratch lives inside the clone and is removed after the
process exits; scratch left by a killed process is removed by the next Enola run
on that clone. A local checkout passed as a path is the caller's own: it is not
measured and not used for scratch.
All repositories sharing these roots count toward the same measured budget.
The storage is scanned once per add; each document copy then checks its known
size on disk against that baseline before writing, reserving room for the temporary file
even when replacing identical stored content. Existing durable content-addressed
documents are not rolled back on failure because other datasets can share those objects.
Concurrent storage writers can leave successfully stored objects from an
interrupted add; this follows the existing ingestion storage contract.

A budget error names the measured usage or free space and the operator limit; a
failure reported by the filesystem itself (no space left, quota exceeded) names
neither.
The HTTP API returns 507 for synchronous failures; background pipeline runs carry
`RepositoryResourceError` in `error_class`. A caller receiving pipeline results
must inspect failed runs before reporting ingestion success.

The guard polls while processes run and waits for owned process cleanup on
failure. Free space is checked every 0.1 s; the size scan pauses ten scan times
between scans (at least 0.1 s). Polling can overshoot, and unrelated writers can
consume free space.
Configure budgets below the deployment's hard quota with enough headroom. This
is not a quota reservation system or automatic cache eviction.

A large repository with mostly excluded assets can need far less local disk.
If the selected content, Git objects and processing output exceed capacity, use
a worker with more disk. This does not provide a 50 GiB ingestion guarantee on
a 4 GiB volume. Cloud worker sizing, token budgets and billing remain separate.
