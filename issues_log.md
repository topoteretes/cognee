# MemoryOS Project Issues & Tracking Board

This document serves as our open-source issue registry to track critical hackathon enhancements, assignments, and reviewer approvals.

---

### 🔴 Issue #101: Scoped Multi-Session Memory (Session ID Isolation)
* **Type:** `enhancement` | `hackathon` | `critical`
* **Status:** `CLOSED (Resolved by Commit a867613)`
* **Assignee/Owner:** Lead Developer
* **Reviewer:** Principal Architect
* **Description:** Cognee v1.2.2 supports temporary session-scoped memory caching (`session_id`). Currently, MemoryOS uses a single global dataset. We need to expose a `session_id` input in the Ingestion and Recall UIs, update the FastAPI schemas, and pass the parameter down to Cognee to isolate memories by chat/agent session.

---

### 🟡 Issue #102: Interactive Tenant API Key Binding (Access Control)
* **Type:** `security` | `hackathon` | `high`
* **Status:** `CLOSED (Resolved by Commit a867613)`
* **Assignee/Owner:** Security Engineer
* **Reviewer:** Principal Architect
* **Description:** The FastAPI backend enforces tenant checking via `X-Tenant-Auth` headers, but the frontend lacks a configuration setting to test or input this live. Add a "Tenant API Key" field in the System Settings tab, persist it in `localStorage`, and inject it into all frontend fetch requests.

---

### 🟢 Issue #103: Raw Text Ingestion Token Size Estimator & Caution Banner
* **Type:** `ui-polish` | `ux` | `medium`
* **Status:** `CLOSED (Resolved by Commit a867613)`
* **Assignee/Owner:** UI Architect
* **Reviewer:** Lead Designer
* **Description:** Ingesting large blocks of text can silently exceed Groq's TPM limits and trigger long retries, making the app feel like it has crashed. Add a live token size estimator (character count / 4) below the text input area and display a warning banner if it exceeds 1,500 tokens to notify users of rate limit retries.

---

### 🔵 Issue #104: Unified Runtime Health Dashboard for Memory Pipeline Monitoring
* **Type:** `enhancement` | `observability` | `medium`
* **Status:** `CLOSED (Resolved by Commit d91a26b)`
* **Assignee/Owner:** Lead Developer
* **Reviewer:** Principal Architect
* **Description:** Cognee lacks a centralized dashboard showing connections, provider status, datasets, and operational statistics (nodes, edges, last recall/improve timestamps). We need to build a backend REST endpoint `/api/health/runtime` aggregating all telemetry details, and a responsive frontend dashboard tab displaying standard GUI indicators, a CLI simulated output table, and a raw JSON explorer.


