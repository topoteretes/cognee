# ⚠️ DEPRECATED - Go to `new-examples/` Instead

This folder is deprecated. All examples have been reorganized into `/new-examples/`.

## Migration Guide

| Old Location | New Location |
|--------------|--------------|
| `python/simple_example.py` | `new-examples/demos/simple_default_cognee_pipelines_example.py` |
| `python/cognee_simple_document_demo.py` | `new-examples/demos/simple_document_qa/` |
| `python/multimedia_example.py` | `new-examples/demos/multimedia_processing/` |
| `python/ontology_demo_example.py` | `new-examples/demos/ontology_reference_vocabulary/` |
| `python/ontology_demo_example_2.py` | `new-examples/demos/ontology_medical_comparison/` |
| `python/temporal_example.py` | `new-examples/demos/temporal_awareness_example.py` |
| `python/conversation_session_persistence_example.py` | `new-examples/demos/conversation_session_persistence_example.py` |
| `python/simple_node_set_example.py` | `new-examples/demos/nodeset_memory_grouping_with_tags_example.py` |
| `python/weighted_edges_example.py` | `new-examples/demos/weighted_edges_relationships_example.py` |
| `python/dynamic_multiple_edges_example.py` | `new-examples/demos/dynamic_multiple_weighted_edges_example.py` |
| `python/web_url_fetcher_example.py` | `new-examples/demos/web_url_content_ingestion_example.py` |
| `python/permissions_example.py` | `new-examples/configurations/permissions_example/` |
| `python/run_custom_pipeline_example.py` | `new-examples/custom_pipelines/custom_cognify_pipeline_example.py` |
| `python/dynamic_steps_example.py` | `new-examples/custom_pipelines/dynamic_steps_resume_analysis_hr_example.py` |
| `python/memify_coding_agent_example.py` | `new-examples/custom_pipelines/memify_coding_agent_rule_extraction_example.py` |
| `python/agentic_reasoning_procurement_example.py` | `new-examples/custom_pipelines/agentic_reasoning_procurement_example.py` |
| `python/code_graph_example.py` | `new-examples/custom_pipelines/code_graph_repository_analysis_example.py` |
| `python/relational_database_migration_example.py` | `new-examples/custom_pipelines/relational_database_to_knowledge_graph_migration_example.py` |
| `database_examples/chromadb_example.py` | `new-examples/configurations/database_examples/chromadb_vector_database_configuration.py` |
| `database_examples/kuzu_example.py` | `new-examples/configurations/database_examples/kuzu_graph_database_configuration.py` |
| `database_examples/neo4j_example.py` | `new-examples/configurations/database_examples/neo4j_graph_database_configuration.py` |
| `database_examples/neptune_analytics_example.py` | `new-examples/configurations/database_examples/neptune_analytics_aws_database_configuration.py` |
| `database_examples/pgvector_example.py` | `new-examples/configurations/database_examples/pgvector_postgres_vector_database_configuration.py` |
| `low_level/pipeline.py` | `new-examples/custom_pipelines/organizational_hierarchy/` |
| `low_level/product_recommendation.py` | `new-examples/custom_pipelines/product_recommendation/` |
| `start_ui_example.py` | `new-examples/demos/start_local_ui_frontend_example.py` |
| `relational_db_with_dlt/relational_db_and_dlt.py` | `new-examples/custom_pipelines/relational_database_to_knowledge_graph_migration_example.py` |


## Files NOT Migrated

| File | Reason |
|------|--------|
| `python/graphiti_example.py` | External Graphiti integration; not core Cognee |
| `python/weighted_graph_visualization.html` | Generated artifact, not source code |
| `python/artifacts/` | Output directory, not example code |
| `relational_db_with_dlt/fix_foreign_keys.sql` | SQL helper script, not standalone example |
| `python/ontology_input_example/` | Data files moved to ontology demo folders |
| `low_level/*.json` | Data files moved to respective pipeline folders |
