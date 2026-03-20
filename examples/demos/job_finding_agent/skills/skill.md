# Job Evaluation Skill

## Purpose
Evaluate incoming job descriptions against the candidate profile and return a practical
recommendation (`APPLY` or `DONT_APPLY`) with concise reasoning.

## Context Source
- Candidate profile: `data/cv.md`
- Mocked job/feedback stream: `data/mock_jobs.json`
- Runtime memory: Cognee dataset/session state

## Operating Rules
- Prioritize roles that match NLP/retrieval/production AI strengths.
- Prefer roles with measurable engineering ownership over purely non-technical positions.
- Be explicit when fit is weak due to domain, seniority, or function mismatch.
- Keep rationale concrete and grounded in job requirements.

## Decision Output Contract
- Decision: `APPLY` or `DONT_APPLY`
- Confidence: float in `[0, 1]`
- Rationale: short, evidence-based explanation

## Feedback Assimilation
- Use selected mocked feedback for the chosen recommendation branch.
- Convert feedback into concise new heuristics.
- Append updates without deleting previous useful rules.
