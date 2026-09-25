-- Relational source for the company brain guide: Acorn Analytics' HR and project database.
-- Regenerate the database with:
--   rm -f examples/demos/company_brain/multi_source/data/company.db
--   sqlite3 examples/demos/company_brain/multi_source/data/company.db < examples/demos/company_brain/multi_source/data/schema.sql
-- (company_brain.py rebuilds it from this file with Python's sqlite3 when it is missing.)
--
-- The tables are normalized. The three *_profiles views join them into one readable
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
