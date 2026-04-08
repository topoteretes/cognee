"""Cognee feature catalog — the product being sold."""

from __future__ import annotations

from enum import Enum


class CogneeFeature(str, Enum):
    MULTIMODAL_INGESTION = "multimodal_ingestion"
    KNOWLEDGE_STRUCTURING = "knowledge_structuring"
    ACCESS_CONTROL = "access_control"
    RETRIEVAL = "retrieval"
    MEMORY = "memory"
    FEEDBACK = "feedback"


FEATURE_DESCRIPTIONS: dict[CogneeFeature, dict] = {
    CogneeFeature.MULTIMODAL_INGESTION: {
        "name": "Multi-modal Ingestion",
        "tagline": "Unified data layer across PDFs, APIs, databases, and more",
        "angles": {
            "developer_experience": "Single SDK call to ingest from 20+ data sources",
            "compliance": "Full audit trail for every data source ingested into the graph",
            "roi": "Replace 3-4 separate ingestion tools with one unified layer",
            "simplicity": "Connect any data source in minutes — no ETL pipelines needed",
            "research": "Ingest papers, datasets, and experimental logs into one knowledge graph",
        },
    },
    CogneeFeature.KNOWLEDGE_STRUCTURING: {
        "name": "Knowledge Structuring",
        "tagline": "Ontology grounding and custom graph models for structured knowledge",
        "angles": {
            "developer_experience": "Define custom Pydantic models and Cognee builds the graph automatically",
            "compliance": "Ground your knowledge in standardized ontologies for regulatory alignment",
            "roi": "Structured knowledge means fewer hallucinations and more accurate answers",
            "simplicity": "Automatic entity extraction — no manual graph building required",
            "research": "Map your domain ontology to a knowledge graph for reproducible analysis",
        },
    },
    CogneeFeature.ACCESS_CONTROL: {
        "name": "Access Control & Isolation",
        "tagline": "Tenant isolation with user, team, and scope-level permissions",
        "angles": {
            "developer_experience": "Built-in multi-tenant isolation — one line to scope queries per user",
            "compliance": "SOC2-friendly ACLs with per-dataset read/write/delete permissions",
            "roi": "Serve multiple customers from one deployment without data leakage risk",
            "simplicity": "Enable isolation with a single environment variable",
            "research": "Each research group gets their own isolated knowledge graph",
        },
    },
    CogneeFeature.RETRIEVAL: {
        "name": "Retrieval",
        "tagline": "Semantic, graph-based, and temporal search over your knowledge",
        "angles": {
            "developer_experience": "10+ search types via one SDK call — graph, vector, temporal, hybrid",
            "compliance": "Permission-filtered retrieval ensures users only see authorized data",
            "roi": "Graph-augmented retrieval outperforms pure vector RAG on complex queries",
            "simplicity": "Replace your brittle RAG pipeline with a single cognee.search() call",
            "research": "Temporal search lets you query how knowledge evolved over time",
        },
    },
    CogneeFeature.MEMORY: {
        "name": "Memory",
        "tagline": "Long-term, short-term, and procedural memory for AI agents",
        "angles": {
            "developer_experience": "Add persistent memory to any agent with a single decorator",
            "compliance": "Auditable memory traces — see exactly what context influenced each decision",
            "roi": "Agents that remember reduce repeat work and improve over time",
            "simplicity": "Memory just works — no vector DB tuning or retrieval pipeline setup",
            "research": "Personalized memory per researcher — each person's queries build their own context",
        },
    },
    CogneeFeature.FEEDBACK: {
        "name": "Feedback for Self-improvement",
        "tagline": "Implicit, explicit, and outcome-driven feedback loops for agents",
        "angles": {
            "developer_experience": "Feedback traces are saved as graph nodes — query them like any other data",
            "compliance": "Every feedback signal is traceable to its source for audit purposes",
            "roi": "Self-improving agents reduce manual tuning and maintenance costs over time",
            "simplicity": "Wrap your agent function with a decorator — feedback collection is automatic",
            "research": "Track how agent performance improves across feedback iterations",
        },
    },
}
