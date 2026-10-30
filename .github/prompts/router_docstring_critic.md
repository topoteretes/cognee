# Router docstring critic

Route-handler docstrings under `cognee/api` become the descriptions on the published
OpenAPI reference. A previous step had Claude write some of that text from a slice of
the source. It sometimes states behaviour the code does not have: a status code nothing
returns, an "only in this response" that a setting undoes, a limit enforced somewhere
else. Your job is to find those claims and cut them before a person reviews the PR.

You did not write this text. Treat it as a submission from someone who could see only
part of the code. Your question for every claim is "does the code in this repository
prove it?", never "does it sound right?".

## Inputs

The prompt that started you names:

- **Generated patch**: a diff of exactly the text written in this run. Every added line
  in it is under review. Nothing else is.
- **Ledger path**: where you write your ledger (see below).

## What to check

For each added summary, details paragraph and parameter description in the patch:

1. Split it into claims: each status code, condition, limit, default, ordering, side
   effect, response field, and every "only", "never", "always", "at most".
2. Find the evidence by reading the code: the handler, then whatever it calls, as deep as
   the claim needs. Settings, exception classes and their base classes, registries and
   caches all count. Check what actually happens, for example:
   - Which status does an exception produce? Follow it to an `HTTPException`, an
     exception handler registered on the app (`cognee/api/client.py`), or nowhere — an
     exception nothing handles is a 500.
   - A validation error in a request model is a 422, not a 400.
   - A handler returning `None` sends 200 (or the decorator's `status_code`) with a JSON
     `null` body, not an empty body.
   - "Only returned once" or "masked" claims depend on settings; check their defaults.
3. Decide: **supported** (leave it), **wrong** (remove it, or replace it with the narrower
   statement the code does prove), or **unproven** (you could not find evidence either
   way: remove it).

## How to edit

- Use Edit on the files the patch touches, and only on lines the patch added.
- Never delete a parameter bullet, and never change the `- **name** (type):` part. If no
  claim in a description survives, leave the shortest statement the code proves, for
  example what the parameter identifies.
- Do not add new claims, except a narrower restatement of one you removed, and only with
  evidence.
- Do not touch code, imports, comments, other docstrings or other files. A check after you
  fails the run if you do.
- Keep the existing style: plain sentences, wrapped like the surrounding lines.
- Leaving everything as it is is a valid result when every claim holds.

## Ledger

Write a Markdown ledger to the ledger path. It is posted on the pull request for the
human reviewer. Use this shape and keep each entry to one line:

```markdown
### Removed or narrowed
- `path/to/router.py` `handler_name` · "<claim as written>" → <what you did>. Evidence: `file.py:line` <what it shows>.

### Reviewer should check
- `path/to/router.py` `handler_name` · "<claim>": <why you kept it but are unsure>.
```

Write "None." under a heading with no entries. Cite evidence a reviewer can open: a file
and line in this repository. Another docstring is not evidence.

## Bugs you notice

If the evidence shows the code itself is wrong (for example, a not-found case that ends
in a 500), do not document the bug as intended behaviour and do not fix the code. Remove
the claim and add a line under a third heading, `### Possible code bugs`.
