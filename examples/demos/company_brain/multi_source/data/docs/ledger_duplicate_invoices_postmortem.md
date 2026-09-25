# Postmortem: duplicate invoice emails for Kestrel Bank

Date of incident: 9 September 2026
Ticket: T-1043
Author: Omar Haddad

## Summary

For about four hours, Kestrel Bank customers received each invoice email two or
three times. No invoice was charged twice; only the emails were duplicated.

## Root cause

The Ledger email worker retried a send whenever the mail provider answered slowly,
even when the first send had succeeded. A provider slowdown that morning turned
every slow answer into a duplicate email.

## Resolution

Lena Fischer made the email worker record each sent invoice id and skip ids it has
already sent. The fix shipped the same afternoon and T-1043 was resolved.

## Follow-ups

- Sam Okafor added a Beacon alert for more than one email per invoice id.
- Omar Haddad will review every Ledger worker that retries external calls.
- Tomas Novak confirmed with Kestrel Bank that no customer was charged twice.
