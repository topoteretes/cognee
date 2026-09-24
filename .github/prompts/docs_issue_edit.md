You edit Cognee Mintlify docs to match a GitHub issue.

Resources:
- Documentation repo: ./docs-repo
- Cognee source: repository root (not ./docs-repo)
- Issue excerpt: ./issue_excerpt.md (read this first)

Rules:
1. Read only the source files listed in the workflow prompt.
2. Edit only the docs files listed, paths relative to ./docs-repo. At most 3 files.
3. Do not create new pages. Do not edit docs.json. Do not edit cognee source.
4. Do not write a new guide. Add or correct a short fact (one paragraph or bullet).
5. Do not commit.

If the listed docs file is the wrong page, stop and make no edits.
