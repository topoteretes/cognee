---
name: code-review
description: >
  Review code for bugs, style issues, and improvements.
  Use when the user asks to "review this code", "find bugs",
  "check for issues", "improve this function", or submits
  a pull request or code snippet for feedback.
---

# Code Review

This skill reviews code for correctness, style, performance, and security issues.

## When to Activate

- User submits code and asks for a review
- User asks to find bugs or issues in code
- User wants improvement suggestions for a function or module

## Process

1. Read the code and identify the language
2. Check for:
   - Correctness bugs (off-by-one, null refs, race conditions)
   - Security issues (injection, hardcoded secrets, unsafe deserialization)
   - Performance concerns (N+1 queries, unnecessary allocations)
   - Style and readability (naming, complexity, documentation)
3. Produce a structured review

## Output Format

```
## Summary
[Overall assessment: looks good / needs minor fixes / has critical issues]

## Issues Found
### Critical
- [issue + fix suggestion]

### Improvements
- [suggestion]

## What Looks Good
- [positive feedback]
```

## Guidelines

1. Always explain *why* something is an issue, not just *what*
2. Suggest concrete fixes, not vague advice
3. Acknowledge good patterns -- reviews should not be only negative
4. Prioritize: security > correctness > performance > style
