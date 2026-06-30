# Cognee Helm Chart

Deploys the Cognee backend with a bundled PostgreSQL + pgvector database.

## Prerequisites

- Kubernetes 1.25+
- Helm 3.10+
- `kubectl` configured for your cluster

---

## Install

### Development (inline credentials)

```bash
helm upgrade --install cognee deployment/helm \
  --namespace cognee --create-namespace \
  --set postgres.auth.password="changeme" \
  --set cognee.llmProvider="openai" \
  --set cognee.llmModel="openai/gpt-4o-mini"
```

> **Note:** Without `existingSecret`, the chart creates a Secret from `postgres.auth.password`.  
> `LLM_API_KEY` will be empty — patch it manually or use `existingSecret` instead.

### Production (recommended)

Create the Secret outside Helm (kubectl, External Secrets, Vault, etc.):

```bash
kubectl create secret generic cognee-credentials \
  --namespace cognee \
  --from-literal=LLM_API_KEY="sk-..." \
  --from-literal=DB_PASSWORD="strongpassword"
```

Then install referencing it:

```bash
helm upgrade --install cognee deployment/helm \
  --namespace cognee --create-namespace \
  --set existingSecret="cognee-credentials" \
  --set postgres.auth.password=""
```

The Deployment and Postgres both read credentials from this Secret.  
Rotating the Secret triggers an automatic rolling restart via checksum annotations.

---

## Upgrade

```bash
helm upgrade cognee deployment/helm --namespace cognee
```

---

## Uninstall

```bash
helm uninstall cognee --namespace cognee
```

---

## Scaling

`replicaCount` is configurable, but the repository currently contains process-local
LRU caches, asyncio locks, and semaphores, and documents single-worker assumptions
in `session_lock.py`. Scaling beyond one replica should be validated against the
application's distributed coordination requirements before use in production.

---

## Values

| Key | Default | Description |
|-----|---------|-------------|
| `replicaCount` | `1` | Number of Cognee replicas. See Scaling note above. |
| `image.repository` | `cognee/cognee` | Cognee image repository |
| `image.tag` | `main` | Image tag |
| `image.pullPolicy` | `IfNotPresent` | Image pull policy |
| `service.type` | `ClusterIP` | `ClusterIP`, `NodePort`, or `LoadBalancer` |
| `service.port` | `8000` | Service port |
| `cognee.env` | `local` | Runtime environment (`ENV`) |
| `cognee.llmProvider` | `openai` | LLM provider |
| `cognee.llmModel` | `openai/gpt-4o-mini` | LLM model |
| `cognee.vectorDbProvider` | `pgvector` | Vector database provider |
| `cognee.enableBackendAccessControl` | `false` | Enable multi-tenant access control |
| `existingSecret` | `""` | Name of existing Secret with `LLM_API_KEY` and `DB_PASSWORD` |
| `resources.requests.cpu` | `500m` | CPU request |
| `resources.requests.memory` | `512Mi` | Memory request |
| `resources.limits.cpu` | `4000m` | CPU limit |
| `resources.limits.memory` | `2Gi` | Memory limit |
| `serviceAccount.create` | `true` | Create a dedicated ServiceAccount |
| `serviceAccount.name` | `""` | Override ServiceAccount name |
| `serviceAccount.automountServiceAccountToken` | `false` | Mount API token into pods |
| `podSecurityContext` | `{}` | Pod-level security context |
| `securityContext.allowPrivilegeEscalation` | `false` | Prevent privilege escalation |
| `securityContext.capabilities.drop` | `[ALL]` | Drop Linux capabilities |
| `startupProbe.enabled` | `true` | Guard against traffic before migration completes |
| `startupProbe.failureThreshold` | `30` | Attempts before pod is failed |
| `startupProbe.periodSeconds` | `10` | Seconds between probe attempts |
| `readinessProbe.enabled` | `true` | Remove pod from Service when dependencies are unhealthy |
| `readinessProbe.initialDelaySeconds` | `10` | Delay before first readiness check |
| `readinessProbe.periodSeconds` | `10` | Seconds between readiness checks |
| `livenessProbe.enabled` | `false` | Disabled — see note below |
| `postgres.image.repository` | `pgvector/pgvector` | Postgres image |
| `postgres.image.tag` | `pg17` | Postgres image tag |
| `postgres.port` | `5432` | Postgres port |
| `postgres.auth.username` | `cognee` | Postgres username |
| `postgres.auth.password` | `""` | Postgres password (dev only; use `existingSecret` in production) |
| `postgres.auth.database` | `cognee_db` | Postgres database name |
| `postgres.storage` | `2Gi` | PVC size for Postgres data |
| `postgres.resources.requests.cpu` | `250m` | Postgres CPU request |
| `postgres.resources.requests.memory` | `256Mi` | Postgres memory request |
| `postgres.resources.limits.cpu` | `1000m` | Postgres CPU limit |
| `postgres.resources.limits.memory` | `1Gi` | Postgres memory limit |

---

## Liveness Probe

`livenessProbe` is disabled by default. The repository's `/health` endpoint verifies
external dependencies (database, vector store, graph store, filesystem). Wiring it
to liveness causes `CrashLoopBackOff` during transient dependency failures — the pod
restarts when the database is temporarily unavailable, preventing natural recovery.

Enable liveness only when the repository exposes a process-only health endpoint
(e.g. `/live`) that answers "is the process alive?" independently of external systems.

---

## Access

```bash
kubectl port-forward svc/cognee-cognee-chart -n cognee 8000:8000
```

API available at `http://localhost:8000`.
