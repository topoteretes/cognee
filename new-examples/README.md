# Cognee Examples

## 📁 Structure

| Folder | Purpose |
|--------|---------|
| `configurations/` | Database, LLM, embedding, and permission setups |
| `custom_pipelines/` | Building custom memory pipelines |
| `demos/` | Feature demos and getting started examples |

## 🔧 Configurations

| Path | Description |
|------|-------------|
| `database_examples/chromadb_vector_database_configuration.py` | ChromaDB vector database |
| `database_examples/kuzu_graph_database_configuration.py` | KuzuDB graph database |
| `database_examples/neo4j_graph_database_configuration.py` | Neo4j graph database |
| `database_examples/neptune_analytics_aws_database_configuration.py` | AWS Neptune Analytics |
| `database_examples/pgvector_postgres_vector_database_configuration.py` | PostgreSQL with PGVector |
| `database_examples/s3_storage_configuration.py` | Amazon S3 storage |
| `llm_configurations/openai_setup.py` | OpenAI LLM setup |
| `llm_configurations/azure_openai_setup.py` | Azure OpenAI LLM setup |
| `embedding_configurations/openai_setup.py` | OpenAI embeddings |
| `embedding_configurations/azure_openai_setup.py` | Azure OpenAI embeddings |
| `structured_output_configurations.py/baml_setup.py` | BAML structured output |
| `structured_output_configurations.py/litellm_intructor_setup.py` | LiteLLM Instructor setup |
| `permissions_example/` | Multi-user access control (with sample PDF) |
| `distributed_execution_with_modal_example.py` | Scale with Modal.com |

## 🔄 Custom Pipelines

| Path | Description |
|------|-------------|
| `custom_cognify_pipeline_example.py` | Customize cognify pipelines |
| `memify_coding_agent_rule_extraction_example.py` | Extract rules from conversations |
| `relational_database_to_knowledge_graph_migration_example.py` | SQL to knowledge graph |
| `agentic_reasoning_procurement_example.py` | AI procurement assistant |
| `code_graph_repository_analysis_example.py` | Code repository analysis |
| `dynamic_steps_resume_analysis_hr_example.py` | CV/resume filtering |
| `organizational_hierarchy/` | Org structure graphs (with JSON data) |
| `organizational_hierarchy/organizational_hierarchy_pipeline_low_level_example.py` | Low-level pipeline variant |
| `product_recommendation/` | Recommendation system (with customer data) |

## 🎯 Demos

| Path | Description |
|------|-------------|
| `simple_default_cognee_pipelines_example.py` | Default pipeline usage ★ |
| `simple_document_qa/` | Document Q&A (with alice_in_wonderland.txt) |
| `core_features_getting_started_example.py` | Intro to Cognee |
| `multimedia_processing/` | Audio/image processing (with media files) |
| `ontology_reference_vocabulary/` | Ontology as vocabulary (with OWL file) |
| `ontology_medical_comparison/` | Medical ontology comparison (with papers + OWL) |
| `web_url_content_ingestion_example.py` | Extract from web pages and ingest directly to memory |
| `temporal_awareness_example.py` | Time-based queries |
| `retrievers_and_search_examples.py` | Retriever patterns and search types guide |
| `nodeset_memory_grouping_with_tags_example.py` | Memory grouping with tags |
| `weighted_edges_relationships_example.py` | Weighted edge relationships |
| `dynamic_multiple_weighted_edges_example.py` | Multiple weighted edges |
| `custom_graph_model_entity_schema_definition.py` | Custom entity schemas ★ |
| `graph_visualization_example.py` | Visualize knowledge graphs |
| `conversation_session_persistence_example.py` | Session persistence |
| `custom_prompt_guide.py` | Custom prompts for extraction |
| `direct_llm_call_for_structured_output_example.py` | Direct LLM structured output |
| `start_local_ui_frontend_example.py` | Launch Cognee UI |
