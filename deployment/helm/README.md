# Cognee Helm Chart

Deploys the Cognee backend with optional bundled PostgreSQL + pgvector.

## Prerequisites

- Kubernetes 1.25+
- Helm 3.10+
- `kubectl` configured for your cluster

---

## Install

### Development (bundled postgres, inline credentials)

```bash
helm upgrade --install cognee deployment/helm \
  --namespace cognee --create-namespace \
  --set postgres.auth.password="changeme"
```

### Production (recommended)

```bash
kubectl create secret generic cognee-credentials \
  --namespace cognee \
  --from-literal=LLM_API_KEY="sk-..." \
  --from-literal=DB_PASSWORD="strongpassword"

helm upgrade --install cognee deployment/helm \
  --namespace cognee --create-namespace \
  --set existingSecret="cognee-credentials"
```

### Production with external managed database

```bash
helm upgrade --install cognee deployment/helm \
  --namespace cognee --create-namespace \
  --set existingSecret="cognee-credentials" \
  --set postgres.enabled=false \
  --set externalDatabase.host="my-db.rds.amazonaws.com" \
  --set externalDatabase.database="cognee_db" \
  --set externalDatabase.username="cognee"
```

### Production with Ingress + TLS

```bash
helm upgrade --install cognee deployment/helm \
  --namespace cognee --create-namespace \
  --set existingSecret="cognee-credentials" \
  --set ingress.enabled=true \
  --set ingress.className=nginx \
  --set "ingress.hosts[0].host=cognee.example.com" \
  --set "ingress.hosts[0].paths[0].path=/" \
  --set "ingress.tls[0].secretName=cognee-tls" \
  --set "ingress.tls[0].hosts[0]=cognee.example.com"
```

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
LRU caches, asyncio locks, and semaphores documented in `session_lock.py`. Validate
distributed coordination requirements before scaling beyond one replica in production.

---

## Values

### Core

| Key | Default | Description |
|-----|---------|-------------|
| `replicaCount` | `1` | Replicas. See Scaling note. |
| `image.repository` | `cognee/cognee` | Image repository |
| `image.tag` | `main` | Image tag |
| `image.pullPolicy` | `IfNotPresent` | Pull policy |
| `existingSecret` | `""` | Secret name with `LLM_API_KEY` and `DB_PASSWORD` |

### Application

| Key | Default | Description |
|-----|---------|-------------|
| `cognee.env` | `local` | `ENV` environment value |
| `cognee.llmProvider` | `openai` | LLM provider |
| `cognee.llmModel` | `openai/gpt-4o-mini` | LLM model |
| `cognee.vectorDbProvider` | `pgvector` | Vector DB provider |
| `cognee.enableBackendAccessControl` | `false` | Multi-tenant isolation |

### Networking

| Key | Default | Description |
|-----|---------|-------------|
| `service.type` | `ClusterIP` | `ClusterIP`, `NodePort`, or `LoadBalancer` |
| `service.port` | `8000` | Service port |
| `ingress.enabled` | `false` | Enable Ingress |
| `ingress.className` | `""` | Ingress class (nginx, traefik, alb, etc.) |
| `ingress.annotations` | `{}` | Ingress annotations |
| `ingress.hosts` | `[{host: cognee.local, paths: [{path: /}]}]` | Host routing |
| `ingress.tls` | `[]` | TLS configuration. References existing Secrets only. |

### Resources

| Key | Default | Description |
|-----|---------|-------------|
| `resources.requests.cpu` | `500m` | CPU request |
| `resources.requests.memory` | `512Mi` | Memory request |
| `resources.limits.cpu` | `4000m` | CPU limit |
| `resources.limits.memory` | `2Gi` | Memory limit |

### Security

| Key | Default | Description |
|-----|---------|-------------|
| `serviceAccount.create` | `true` | Create dedicated ServiceAccount |
| `serviceAccount.automountServiceAccountToken` | `false` | Mount API token |
| `podSecurityContext` | `{}` | Pod security context |
| `securityContext.allowPrivilegeEscalation` | `false` | Prevent privilege escalation |
| `securityContext.capabilities.drop` | `[ALL]` | Drop Linux capabilities |
| `networkPolicy.enabled` | `false` | Enable NetworkPolicy |
| `networkPolicy.ingress` | `[]` | Ingress rules (allow-all if empty) |
| `networkPolicy.egress` | `[]` | Egress rules (allow-all if empty) |

### Availability

| Key | Default | Description |
|-----|---------|-------------|
| `autoscaling.enabled` | `false` | Enable HPA (transfers replica ownership from Deployment) |
| `autoscaling.minReplicas` | `2` | HPA minimum replicas |
| `autoscaling.maxReplicas` | `5` | HPA maximum replicas |
| `autoscaling.targetCPUUtilizationPercentage` | `80` | CPU target |
| `podDisruptionBudget.enabled` | `false` | Enable PDB (only renders when replicaCount > 1) |
| `podDisruptionBudget.maxUnavailable` | `1` | Max pods unavailable during disruption |

### Probes

| Key | Default | Description |
|-----|---------|-------------|
| `startupProbe.enabled` | `true` | Guards traffic until Alembic migration completes |
| `startupProbe.failureThreshold` | `30` | Max attempts |
| `startupProbe.periodSeconds` | `10` | Seconds between attempts |
| `readinessProbe.enabled` | `true` | Removes pod from Service when dependencies unhealthy |
| `readinessProbe.initialDelaySeconds` | `10` | Initial delay |
| `readinessProbe.periodSeconds` | `10` | Period |
| `livenessProbe.enabled` | `false` | See note below |

### PostgreSQL (bundled)

| Key | Default | Description |
|-----|---------|-------------|
| `postgres.enabled` | `true` | Deploy bundled PostgreSQL StatefulSet |
| `postgres.image.repository` | `pgvector/pgvector` | Image |
| `postgres.image.tag` | `pg17` | Tag |
| `postgres.port` | `5432` | Port |
| `postgres.auth.username` | `cognee` | Username |
| `postgres.auth.password` | `""` | Password (dev only; use `existingSecret`) |
| `postgres.auth.database` | `cognee_db` | Database name |
| `postgres.storage` | `2Gi` | PVC size |
| `postgres.resources.requests.cpu` | `250m` | CPU request |
| `postgres.resources.requests.memory` | `256Mi` | Memory request |
| `postgres.resources.limits.cpu` | `1000m` | CPU limit |
| `postgres.resources.limits.memory` | `1Gi` | Memory limit |

### External Database

Used when `postgres.enabled=false`.

| Key | Default | Description |
|-----|---------|-------------|
| `externalDatabase.host` | `""` | Database host |
| `externalDatabase.port` | `5432` | Database port |
| `externalDatabase.database` | `""` | Database name |
| `externalDatabase.username` | `""` | Database username |
| `externalDatabase.existingSecret` | `""` | Secret with credentials |

---

## Liveness Probe

Disabled by default. `/health` verifies external dependencies (DB, vector store, graph
store, filesystem). Wiring it to liveness causes `CrashLoopBackOff` during transient
dependency failures. Enable only when the repository exposes a process-only endpoint.

---

## Access

```bash
kubectl port-forward svc/cognee-cognee-chart -n cognee 8000:8000
```

API available at `http://localhost:8000`.
