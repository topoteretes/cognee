# Remote repository ingestion

Use `add(repo_url, codegraph_config=...)`, then `cognify()` scoped to the data IDs
returned by add. See `examples/guides/remote_code_repository.py` for code-only
processing without an LLM key. Tokens belong in `repo_credentials`, not persisted URLs.

## Download and selection

Remote clones use shallow history, `--filter=blob:none` and an exact sparse
checkout at one commit. The tree is inspected before requesting selected blobs.
Symlinks, submodules, audio/video and known binary artifacts are excluded, and so is
every file inside a directory named `__pycache__`, `build`, `coverage`, `dist`,
`node_modules`, `target` or `venv` (or `.git` or `.enola`) at any depth and in any
letter case. A source package that has such a directory name is therefore left out of
a remote clone's code graph (a local checkout is read whole). A file that merely
carries one of these names is kept. Registered binary documents are included when
`include_documents=True`. Code-only mode retains text and tracked configuration,
including unknown extensions used by custom extractors. There is no file-size cutoff.
The ordinary document ingestion rules still apply after checkout.

Servers that ignore the filter fail rather than downloading a full clone.
Git LFS smudging is disabled. A selected file that is an LFS pointer is not the
file's content, so it is removed from the checkout and counted under
`lfs_pointer` in the manifest's skip counts (the selected file count is of the files
that remain); the rest of the repository is ingested.
Paths containing newlines cannot be represented safely by sparse checkout and
answer 400. After every checkout, files that are not part of the selection are
removed from the work tree.

The manifest records the commit, selection fingerprint, policy version, selected
file count and skip counts. Cognify holds the clone lock and validates both the
commit and selection. If another add changes either, re-add the repository to
that dataset before cognify. A manifest written before partial clones records no
selection and is refused the same way until it is re-added. Documents are copied
to durable storage during add.

A clone without a snapshot record (a full clone from before partial clones, or a
clone whose refresh was interrupted after the record was removed) is replaced under
the same lock after a new partial clone validates. Both coexist during preparation,
so reserve space for that transition. A failure while preparing the replacement
preserves the old cache. The old clone is renamed aside (`.<digest>.old-*`) before
it is deleted, so a process killed meanwhile leaves a directory the next run on that
repository sweeps, not a half-deleted clone. Once retirement starts, cancellation waits
for the validated replacement to be published before propagating; the clone lock stays
held throughout. A cancelled refresh keeps the clone; a refresh stopped by a resource
error removes it to free the space. No unrelated cache or existing dataset graph is
deleted.

## Operator disk settings

Both settings are optional positive integers in bytes, read at operation time. A
variable that is empty counts as unset:

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

A budget error names the measured usage or free space, the operator limit and the
setting to change; a failure reported by the filesystem itself (no space left, quota
exceeded) names neither. Git and Enola report a full disk in different letter case,
and both are recognised.
The HTTP API returns 507 for synchronous failures; background pipeline runs carry
`RepositoryResourceError` in `error_class`. A caller receiving pipeline results
must inspect failed runs before reporting ingestion success.

The guard polls while processes run and waits for owned process cleanup on
failure. Free space is checked every 0.1 s; the size scan pauses ten scan times
between scans (at least 0.1 s). Polling can overshoot, and unrelated writers can
consume free space.
Configure budgets below the deployment's hard quota with enough headroom. This
is not a quota reservation system or automatic cache eviction.

Known limits of the guard, all only with a limit set: every run scans the whole
measured tree (managed clones and local ingestion storage), so the cost grows with the
number of files and with the number of concurrent runs, and the scans share the default
executor; the budget is global to its roots, so any run in flight when the total crosses
the limit is stopped, not necessarily the one that grew; a directory that cannot be read
is skipped; clones, `.tmp-*` and `.old-*` leftovers of repositories that are not touched again
and the rest of the ingestion storage count until they are removed by hand. Checking out a
repository with several hundred thousand selected files is superlinear in git itself and
can reach the 600 s git timeout.

A large repository with mostly excluded assets can need far less local disk.
If the selected content, Git objects and processing output exceed capacity, use
a worker with more disk. This does not provide a 50 GiB ingestion guarantee on
a 4 GiB volume. Cloud worker sizing, token budgets and billing remain separate.
