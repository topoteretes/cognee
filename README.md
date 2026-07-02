# MemoryOS 🧠
**The Control Plane for AI Memory**

![MemoryOS Hero](https://via.placeholder.com/1200x400.png?text=MemoryOS+Control+Plane)

MemoryOS is a diagnostic, observability, and governance layer built for Multi-Agent AI systems. It solves the critical problem of "agent amnesia" and "context corruption" over long time horizons.

> **Technical Honesty Note:** MemoryOS is the *Control Plane*. We rely on the excellent open-source framework [Cognee](https://github.com/topoteretes/cognee) as our *Data Engine* to handle the heavy lifting of graph extraction, vector embeddings, and underlying database storage. MemoryOS adds the UI, Multi-Tenant isolation, Time Machine logging, and deterministic Memory Doctor workflows on top of it.

## 🚀 The Problem
Today's AI agents rely on simple Vector Databases (RAG). Over time, agents ingest contradictory facts (e.g., "John lives in NY" on Monday, "John lives in CA" on Wednesday). Without an observability layer to monitor, detect, and merge these semantic conflicts, the agent's logic breaks down.

## 💡 The Solution
MemoryOS acts as the "Datadog for AI Memory". It provides:
* **Memory Galaxy**: A stunning 3D force-directed topology of the active knowledge graph.
* **Memory Doctor**: An auditing tool that scans the graph for isolated nodes, fuzzy duplicates, and conflicting relational facts.
* **Time Machine**: An event-sourcing ledger that tracks every memory modification over time.

## 🏗️ Architecture
* **Frontend**: Next.js 14, React, Tailwind CSS, React Force Graph.
* **Backend API**: FastAPI, Python 3.10+, JWT Dummy Middleware.
* **Data Engine (Cognee)**: LiteLLM, NetworkX, LanceDB, SQLite.

## 🏁 Quickstart

```bash
# 1. Clone & setup backend
git clone https://github.com/memoryos/hangover.git
cd hangover/backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env # Add your GROQ_API_KEY

# 2. Run backend
uvicorn apps.api.main:app --reload --port 8000

# 3. Run frontend
cd ../apps/web
npm install
npm run dev
```

## ⚖️ Hackathon Implementation Notes
For this demonstration, the backend utilizes local file-based databases (`SQLite` and `LanceDB`) to eliminate cloud dependencies for judges. A `docker-compose.yml` is included in the repository to demonstrate the intended production transition to `PostgreSQL`, `Qdrant`, and `Redis`.
