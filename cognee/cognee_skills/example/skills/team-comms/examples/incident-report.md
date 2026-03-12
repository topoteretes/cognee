# Incident Report

**Severity:** P1
**Date:** March 8, 2025
**Duration:** 47 minutes

## Summary
Payment processing was down for 47 minutes due to a misconfigured database connection pool.

## Timeline
- 14:02 — Alerts fire for payment failures
- 14:08 — On-call engineer begins investigation
- 14:22 — Root cause identified: connection pool maxed out after deploy
- 14:35 — Fix deployed, pool size increased
- 14:49 — All systems nominal, monitoring confirms recovery

## Root Cause
Deploy at 13:45 reduced the connection pool from 50 to 5 (config typo).

## Action Items
- [ ] Add connection pool size to deploy checklist
- [ ] Add alerting for pool exhaustion
- [ ] Review config validation in CI
