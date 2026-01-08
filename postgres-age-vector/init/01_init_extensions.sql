CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

LOAD 'age';
CREATE EXTENSION IF NOT EXISTS age;

GRANT USAGE ON SCHEMA ag_catalog TO PUBLIC;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA ag_catalog TO cognee;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA ag_catalog TO cognee;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA ag_catalog TO cognee;

ALTER DATABASE cognee_db SET search_path = public, ag_catalog, "$user";

DO $$
BEGIN
    RAISE NOTICE 'pgvector: % | AGE: % | search_path: public, ag_catalog',
        (SELECT extversion FROM pg_extension WHERE extname = 'vector'),
        (SELECT extversion FROM pg_extension WHERE extname = 'age');
END $$;
