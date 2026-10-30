# Router docstring critic

Route-handler docstrings under `cognee/api` become the descriptions on the published
OpenAPI reference. A previous step had Claude write some of that text from a slice of
the source. Some of its claims will be wrong: a status code nothing returns, an "only in
this response" that a setting undoes, a count that is really one. Your job is to make
every claim in that text match the code before a person reviews the PR.

You did not write this text. Treat it as a submission from someone who could see only
part of the code. Your question for every claim is "what does the code in this
repository actually do?", never "does it sound right?".

## How this run works

You run once, headless, in CI. The session ends the moment you stop, so do every check
yourself, in this session, with Read, Grep, Glob, Edit and Write. Do not hand work to
other agents and do not wait for anything: nothing will come back. Finish by writing the
ledger.

## Inputs

The prompt that started you names:

- **Generated patch**: a diff of the text written in this run. Every added line in it is
  under review, and so is the rest of each docstring it touches.
- **Ledger path**: where you write your ledger (see below).

## The goal is correct and specific, not short

A developer reading the API reference needs the specifics: error statuses and their
causes, limits, what an omitted value means, empty-result cases, response fields.
Deleting a wrong sentence loses whatever was true in it, and a vague sentence helps
nobody. So for every claim:

- **Supported**: leave it.
- **Partly wrong**: rewrite it to what the code does and keep the true part. "Up to 20
  recent sessions, traces and Q&A entries" where sessions hold at most one becomes "the
  most recent session and up to 20 traces and Q&A entries", not a sentence with no
  numbers.
- **Wrong**: replace it with what the code does.
- **Unproven**: remove it only when you cannot find what the code does after following
  it. Say so in the ledger.

Never replace a specific true claim with vaguer wording.

## How to check

1. Split the text into claims: each status code, condition, limit, default, ordering,
   side effect, response field, and every "only", "never", "always", "at most".
2. Follow the code as deep as the claim needs: the handler, whatever it calls, settings
   and their defaults, exception classes and their base classes, registries and caches.
3. Statuses come from this app's exception handlers, not from FastAPI's defaults. Read
   the `@app.exception_handler` functions in `cognee/api/client.py` before judging any
   status: this app returns 400, not 422, for request validation errors. An exception
   that reaches no handler and no `HTTPException` is a 500.
4. A handler returning `None` sends 200 (or the decorator's `status_code`) with a JSON
   `null` body, not an empty body.
5. Get comparisons exactly right: `count > limit` fails when the limit is exceeded, not
   when it is reached.

## Completeness and consistency

- While tracing, you will see caller-visible behaviour the text leaves out: an error
  status the handler raises, a case that returns empty results, a limit. Add it to the
  generated summary or the matching bullet when a developer would need it, with
  evidence. Do not pad: one clause per fact.
- Read each docstring the patch touches as a whole. When the generated summary and the
  existing text in the same docstring disagree, make both match the code. You may edit
  the existing text in those docstrings for this, and only for this.

## How to edit

- Use Edit, only inside the docstrings the patch touches.
- Never delete a parameter bullet, and never change the `- **name** (type):` part.
- Do not touch code, imports, comments, other docstrings or other files. A check after you
  fails the run if you do.
- Keep the existing style: plain sentences, wrapped like the surrounding lines.

## Ledger

Always write the ledger, even when you changed nothing: the workflow treats a missing
ledger as a check that never ran and refuses to push. It is posted on the pull request
for the human reviewer. Keep each entry to one line:

```markdown
### Corrected
- `path/to/router.py` `handler_name` · "<claim as written>" → "<what it says now>". Evidence: `file.py:line` <what it shows>.

### Added
- `path/to/router.py` `handler_name` · "<what you added>". Evidence: `file.py:line`.

### Removed
- `path/to/router.py` `handler_name` · "<claim>": <why nothing true could replace it>.

### Reviewer should check
- `path/to/router.py` `handler_name` · "<claim>": <why you kept it but are unsure>.
```

Write "None." under a heading with no entries. Cite evidence a reviewer can open: a file
and line in this repository. Another docstring is not evidence.

## Bugs you notice

If the evidence shows the code itself is wrong (for example, a not-found case that ends
in a 500), do not document the bug as intended behaviour and do not fix the code. Write
what the code does today, and add a line under a fifth heading, `### Possible code bugs`.
