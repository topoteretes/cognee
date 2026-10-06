"""Set up the sample company, so the cookbook runs before you point it at your own data.

The sample is Acorn Analytics, a fictional company. This writes its three sources to
sample/ (git-ignored), in the shape of a real company's data:

- company.db: an HR and project database, built from SCHEMA below;
- tickets.json: a support desk export;
- escalations.csv: the escalated tickets Customer Success tracks in a spreadsheet, so the
  run covers a CSV ticket export too;
- docs/: meeting notes, a postmortem and a planning memo.

Some people, projects and customers appear in all three, so the run shows them merge into
one node each. Only LLM_API_KEY is needed; no accounts or data of yours.

Run: uv run python examples/cookbooks/company_brain/company_qa/setup.py
"""

import csv
import json
import shutil
import sqlite3
from pathlib import Path

SAMPLE = Path(__file__).parent / "sample"

# The HR and project database, as SQL. company_qa.py --sample reads its three views.
SCHEMA = """-- The tables are normalized. The three *_profiles views join them into one readable
-- row per entity with `id`, `title` and `content` columns: cognee's dlt document path
-- turns each such row into a text document, so the LLM sees "Dana Kim works in the Search
-- team" instead of a bare team_id foreign key. The CASTs give the view columns a declared
-- type, which dlt needs to map them without a warning.

PRAGMA foreign_keys = ON;

CREATE TABLE teams (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL UNIQUE,
    job_title TEXT NOT NULL,
    team_id INTEGER NOT NULL REFERENCES teams(id),
    manager_id INTEGER REFERENCES employees(id)
);

CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    industry TEXT NOT NULL,
    account_manager_id INTEGER REFERENCES employees(id)
);

CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    summary TEXT NOT NULL,
    team_id INTEGER NOT NULL REFERENCES teams(id),
    customer_id INTEGER REFERENCES customers(id)
);

CREATE TABLE assignments (
    id INTEGER PRIMARY KEY,
    employee_id INTEGER NOT NULL REFERENCES employees(id),
    project_id INTEGER NOT NULL REFERENCES projects(id),
    role TEXT NOT NULL,
    UNIQUE (employee_id, project_id)
);

INSERT INTO teams (id, name) VALUES
    (1, 'Search'),
    (2, 'Billing'),
    (3, 'Platform'),
    (4, 'Customer Success');

INSERT INTO employees (id, full_name, job_title, team_id, manager_id) VALUES
    (1, 'Priya Patel', 'Engineering Director', 3, NULL),
    (2, 'Marco Rossi', 'Engineering Manager', 1, 1),
    (3, 'Dana Kim', 'Senior Software Engineer', 1, 2),
    (4, 'Omar Haddad', 'Engineering Manager', 2, 1),
    (5, 'Lena Fischer', 'Software Engineer', 2, 4),
    (6, 'Sam Okafor', 'Site Reliability Engineer', 3, 1),
    (7, 'Grace Liu', 'Head of Customer Success', 4, NULL),
    (8, 'Tomas Novak', 'Support Engineer', 4, 7);

INSERT INTO customers (id, name, industry, account_manager_id) VALUES
    (1, 'Brightline Retail', 'retail', 7),
    (2, 'Kestrel Bank', 'financial services', 8),
    (3, 'Oakridge Health', 'healthcare', 7);

INSERT INTO projects (id, name, status, summary, team_id, customer_id) VALUES
    (1, 'Atlas', 'active', 'Product search relaunch for the Brightline Retail storefront.', 1, 1),
    (2, 'Ledger', 'active', 'Invoicing and payment reconciliation for Kestrel Bank.', 2, 2),
    (3, 'Beacon', 'active', 'Internal observability and alerting platform.', 3, NULL),
    (4, 'Harbor', 'planned', 'Patient-facing search portal for Oakridge Health.', 1, 3);

INSERT INTO assignments (id, employee_id, project_id, role) VALUES
    (1, 3, 1, 'tech lead'),
    (2, 2, 1, 'engineering manager'),
    (3, 6, 1, 'infrastructure'),
    (4, 5, 2, 'developer'),
    (5, 4, 2, 'engineering manager'),
    (6, 6, 3, 'tech lead'),
    (7, 3, 4, 'tech lead');

CREATE VIEW employee_profiles AS
SELECT
    e.id AS id,
    e.full_name AS title,
    CAST(e.full_name || ' works in the ' || t.name || ' team as ' || e.job_title || '.'
        || COALESCE(' ' || e.full_name || ' reports to ' || m.full_name || '.', '')
        || COALESCE(
            ' ' || e.full_name || ' works on '
            || (SELECT group_concat(p.name || ' as ' || a.role, ' and on ')
                FROM assignments a JOIN projects p ON p.id = a.project_id
                WHERE a.employee_id = e.id)
            || '.',
            ''
        ) AS TEXT) AS content
FROM employees e
JOIN teams t ON t.id = e.team_id
LEFT JOIN employees m ON m.id = e.manager_id;

CREATE VIEW project_profiles AS
SELECT
    p.id AS id,
    p.name AS title,
    CAST(p.name || ' is a project owned by the ' || t.name || ' team. Status: ' || p.status || '. '
        || p.summary
        || COALESCE(' The customer is ' || c.name || '.', ' It is an internal project.') AS TEXT) AS content
FROM projects p
JOIN teams t ON t.id = p.team_id
LEFT JOIN customers c ON c.id = p.customer_id;

CREATE VIEW customer_profiles AS
SELECT
    c.id AS id,
    c.name AS title,
    CAST(c.name || ' is a customer in ' || c.industry || '.'
        || COALESCE(' Its account manager is ' || m.full_name || '.', '') AS TEXT) AS content
FROM customers c
LEFT JOIN employees m ON m.id = c.account_manager_id;
"""

# The support desk export (tickets.json).
TICKETS = [
    {
        "ticket_id": "T-1042",
        "title": "Search autocomplete is slower than 800 ms",
        "customer": "Brightline Retail",
        "project": "Atlas",
        "assignee": "Sam Okafor",
        "status": "in progress",
        "priority": "medium",
        "opened": "2026-09-18",
        "description": "Autocomplete latency on the Brightline Retail storefront rose from 200 ms to over 800 ms during peak hours.",
    },
    {
        "ticket_id": "T-1044",
        "title": "Invoice CSV export is missing the tax column",
        "customer": "Kestrel Bank",
        "project": "Ledger",
        "assignee": "Lena Fischer",
        "status": "open",
        "priority": "low",
        "opened": "2026-09-20",
        "description": "The monthly CSV export no longer includes the VAT amount per invoice.",
    },
    {
        "ticket_id": "T-1045",
        "title": "Single sign-on loops back to the login page on staging",
        "customer": "Oakridge Health",
        "project": "Harbor",
        "assignee": "Tomas Novak",
        "status": "open",
        "priority": "medium",
        "opened": "2026-09-21",
        "description": "Oakridge Health testers are sent back to the login page after signing in to the Harbor staging portal.",
    },
]

# The escalations Customer Success tracks in a spreadsheet (escalations.csv).
ESCALATIONS = [
    {
        "ticket_id": "T-1041",
        "title": "Search results are missing newly added products",
        "customer": "Brightline Retail",
        "project": "Atlas",
        "assignee": "Dana Kim",
        "status": "open",
        "priority": "high",
        "opened": "2026-09-17",
        "description": "Products added to the Brightline Retail catalog after 15 September do not appear in storefront search.",
    },
    {
        "ticket_id": "T-1043",
        "title": "Customers receive duplicate invoice emails",
        "customer": "Kestrel Bank",
        "project": "Ledger",
        "assignee": "Lena Fischer",
        "status": "resolved",
        "priority": "high",
        "opened": "2026-09-09",
        "description": "Kestrel Bank customers received every invoice email two or three times on 9 September.",
    },
]

DOCS = {
    "atlas_weekly_sync_2026-09-19.md": """# Atlas weekly sync — 19 September 2026

Attendees: Marco Rossi, Dana Kim, Sam Okafor, Grace Liu

## Missing products in search (T-1041)

Dana Kim traced the missing products to the Brightline Retail catalog feed. Since
15 September the feed sends new products in a second file, and the Atlas indexer
only reads the first one.

Decision: Dana Kim will change the indexer to read every file in the feed and
re-index the catalog. Target date is 26 September. Until then, Brightline Retail
can trigger a manual re-index from the admin page.

## Autocomplete latency (T-1042)

Sam Okafor found that the autocomplete cache is evicted every time the catalog is
re-indexed. Sam will move the cache to the Beacon-monitored Redis cluster and add
a latency alert.

## Customer

Grace Liu reported that Brightline Retail's contract renewal in October depends on
Atlas search quality. Grace will send Brightline Retail an update after Dana Kim's
fix ships.

## Action items

- Dana Kim: read every catalog feed file, re-index Brightline Retail (due 26 September)
- Sam Okafor: move the autocomplete cache, add a latency alert in Beacon
- Grace Liu: send Brightline Retail a status update
""",
    "ledger_duplicate_invoices_postmortem.md": """# Postmortem: duplicate invoice emails for Kestrel Bank

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
""",
    "q4_planning_memo.md": """# Q4 planning memo

From: Priya Patel
Date: 22 September 2026

## Harbor

Harbor, the patient-facing search portal for Oakridge Health, starts in November.
The Search team owns Harbor. Dana Kim will be the Harbor tech lead once the Atlas
catalog fix for Brightline Retail has shipped, and Marco Rossi stays engineering
manager for both projects.

Tomas Novak is the support contact for Oakridge Health during the Harbor pilot and
is already working on the single sign-on problem on staging (T-1045).

## Ledger

After the duplicate invoice incident, the Billing team spends October on
reliability. Omar Haddad owns the review of Ledger's retry logic, and Lena Fischer
will finish the missing tax column in the CSV export (T-1044).

## Platform

Sam Okafor continues to run Beacon. Every customer-facing project must have Beacon
alerts before it goes live, starting with Harbor.
""",
}


def write_sample() -> None:
    """Write the sources, replacing an earlier sample/ folder."""
    shutil.rmtree(SAMPLE, ignore_errors=True)
    (SAMPLE / "docs").mkdir(parents=True)

    connection = sqlite3.connect(SAMPLE / "company.db")
    connection.executescript(SCHEMA)
    connection.close()

    export = {
        "export": "Acorn Analytics support desk",
        "exported_at": "2026-09-22",
        "tickets": TICKETS,
    }
    (SAMPLE / "tickets.json").write_text(json.dumps(export, indent=2) + "\n")

    with open(SAMPLE / "escalations.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(ESCALATIONS[0]))
        writer.writeheader()
        writer.writerows(ESCALATIONS)

    for name, text in DOCS.items():
        (SAMPLE / "docs" / name).write_text(text)


if __name__ == "__main__":
    write_sample()
    print(
        "[setup] Wrote the sample company to sample/: company.db, tickets.json, escalations.csv, docs/."
    )
    print(
        "[setup] Now run: uv run python "
        "examples/cookbooks/company_brain/company_qa/company_qa.py --sample"
    )
