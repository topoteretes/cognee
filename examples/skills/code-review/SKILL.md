---
name: Code Review
description: Use when reviewing a repository change or codebase slice. Produce concrete findings with file references, severity, impact, and tests.
allowed-tools: memory_search
tags:
  - code-review
  - testing
  - architecture
---

# Code Review Skill

Use this skill when asked to inspect code changes, review a codebase area,
or combine prior agent findings into review feedback.

## Process

1. Start from the concrete review goal. Identify the changed files, API
   surface, or subsystem under review.
2. Recall relevant project memory before making claims. Prefer findings
   that cite a file path, symbol, behavior, or test gap.
3. Look for correctness bugs, permission leaks, state handling mistakes,
   missing validation, weak error handling, and missing tests.
4. Separate verified findings from open questions. Do not turn style
   preferences into review findings unless they create real risk.
5. For every issue, explain the impact and the smallest practical fix.

## Output

Return actionable review findings only. For each finding include:

- Severity: critical, high, medium, or low.
- Location: file path and line, symbol, endpoint, or workflow.
- Problem: what can break or mislead users.
- Fix: the concrete change to make.
- Tests: what test should prove the fix.

If no issues are found, say that directly and list remaining test gaps or
residual risk.
