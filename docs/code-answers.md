# Answers grounded in code

`recall(scope=["graph", "code"])` collects code evidence before generating a
GRAPH_COMPLETION, HYBRID_COMPLETION or RAG_COMPLETION answer. It uses the normal
dataset authorization, session and streaming pipeline. Code-only datasets do
not require document ingestion. The other types that generate or route to an
answer (`GRAPH_COMPLETION_COT`, `GRAPH_COMPLETION_CONTEXT_EXTENSION`,
`GRAPH_COMPLETION_DECOMPOSITION`, `GRAPH_SUMMARY_COMPLETION`, `AGENTIC_COMPLETION`,
`TEMPORAL`, `FEELING_LUCKY` and `BROAD`) are rejected for this combined scope rather
than answering without the requested code evidence. A type chosen by the router
(a question about "2019" routes to `TEMPORAL`) is not the caller's choice: it
answers as `HYBRID_COMPLETION` with the code evidence instead. Retrieval-only types still
return their independent graph and code results.

```python
results = await cognee.recall(
    "Da li pauza konektora blokira ručni sync?",
    datasets=["github_code"],
    scope=["graph", "code"],
)
```

With no `code_query`, an LLM translates the question into English retrieval
terms and selects explanation, callers or impact analysis. Exact identifiers
are retained. Only ids actually retrieved from the graph can seed follow-up
operations. This adds one planning call per retrieval lane before the answer: the
default concurrent session mode runs two lanes (the question and its rewrite from
the conversation), so two calls per turn, and a question is planned once for all
datasets of one search. Session analysis can add its usual calls. Planning
improves multilingual retrieval but does not guarantee that every relevant
function is found.

An explicit `code_query` runs the selected deterministic operation instead of
planning. `only_context=True` also skips planning and generation: supply English
terms or identifiers, or an explicit `code_query`, for a useful preview.
`scope="code"` and `SearchType.CODE` retain their model-free behavior.

## Evidence and source versions

Fresh Enola extraction stores each symbol's own source lines in a `CodeSource`
node of its own, with the line range and the hash of that text; the CodeSymbol
keeps only the hash. A CODE query loads every symbol and never a source node,
so source text costs those queries nothing, and an answer reads the few source
nodes it cites by id. The pinned
Enola reports the line a symbol is declared on, not where it starts or ends, so
both are derived from the file: the start moves up over the decorators,
attributes and comments directly above, and the end is the line before the next
symbol that is not one of its members (a class includes its methods, a Markdown
section its subsections, a C++ scope its members), cut earlier where the
enclosing scope closes or code of the same level follows. These are
approximations of a parser, not a parse: annotations that span lines or share the
line with the declaration are followed, but a declaration split over lines at its
own indent, an unusual layout, or a multi-line string whose lines start at the
declaration's indent can cut a range short. A range from the
extractor wins when it supplies one, and a symbol with no usable range keeps the
whole file. The Git receipt and the snapshot id stay on the repository node
(see the `delta` operation); they are not part of a symbol, and its hash covers
only its own text, so a commit or an edit elsewhere in the file does not rewrite
it. A file rewritten during extraction is rejected, a permission change alone or
a modification time in the future is not. A file outside the repository (a
symlink to elsewhere) and one that is not UTF-8 have no source, and the rest of
the repository is indexed as before.

Every fresh extraction compares each file against the stored graph, because
Enola's snapshot id need not change when a body does. Re-indexing an unchanged
repository therefore reads every referenced file and the stored graph again,
and writes nothing that did not change.

The comparison reads the whole stored graph once per index, as every code graph
load already did, and that read now includes the `CodeSource` nodes. Measured on
a 619 KB module, the source nodes hold about 1.4 times the source in text, and
the whole read serialised as JSON grew from about 5 to about 8 times the source
(a re-index of the unchanged module took under half a second). This only affects
indexing; a CODE query or an answer never loads the whole graph. A large
repository pays it on every re-index. Paths and symlinks
outside the repository root are rejected. Search reads the stored source,
never the current checkout or a moving GitHub branch.

Previously indexed graphs and imported snapshots still work, but have no
stored source. Re-index the repository to capture it. Virtual files and files
that cannot be decoded as UTF-8 have no source evidence. The answer context
identifies nodes without source. A Git receipt with `dirty=true` is not proof
that the source matches the named commit; the stored file hash identifies the
actual captured content.

Answer entries expose `metadata.code_evidence`: graph operation results,
source locations and captured text, missing-source ids (symbols only), and the
question plan. The graph snapshot is cached briefly and the source text is read
when the answer is built, so a symbol re-indexed or removed in between cannot be
cited with the lines of the old index: it is listed in `stale_source_ids`, with
no text, and the model is told not to cite lines for it. A source whose lines lie inside another included source (a
method of an included class) carries `within`, the node id that holds the text,
instead of repeating it.
A `source="code"` entry repeats the graph operations; source locations, captured
text and the question plan are in `metadata.code_evidence` of the answer, and
`search(verbose=True)` returns the same dict as `code_evidence`. JSON and the
final SSE response use the same normalized entries. References are evidence supplied
to the model, not a claim that every generated statement has been mechanically
verified.

The model receives evidence in the context, not as system instructions. Its
instructions require citations, separation of inference from observed code,
and disclosure of relevant missing or truncated coverage. Existing CODE
operation bounds still apply and their coverage flags are retained. There is
no additional character truncation; requests exceeding the model's context
window fail visibly rather than silently dropping evidence. A question that
reaches a large part of the graph can therefore be large: sources are symbol
text, not whole files, and a nested source is not repeated.

Code lookup and planning failures propagate, except a seed the graph cannot
resolve, which becomes a coverage note. A missing text corpus is recorded
as a source state and the answer can use the code evidence. The combined answer
is generated even when nothing matched, from the coverage notes, so it does not
return an empty result for a dataset without matches and recall's
`tools_trigger="on_empty"` does not fire for it. An empty code
match retains its coverage notes, so absence of evidence cannot masquerade as
proof that a function or behavior does not exist.

## Validation

The targeted tests cover source changes, imported snapshots, path escape,
multilingual question planning, callers and impact traversal, code-only and
mixed context, propagation of lookup errors, and normalized evidence. A real
Enola/Ladybug test indexes this SDK's GitHub sync, retirement and signature
verification files and compares retrieved source ranges against those files.
Model responses are mocked in unit tests; these tests do not measure live
answer accuracy or multilingual retrieval quality.
