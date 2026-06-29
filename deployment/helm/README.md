# cognee Helm chart

This chart deploys the cognee backend and, by default, a bundled pgvector Postgres instance.
For production, prefer a managed Postgres service and provide secrets through an existing
Kubernetes Secret instead of storing credentials in Helm values.

## Prerequisites

- Kubernetes 1.25+
- Helm 3
- A cognee image reachable from the cluster
- An LLM API key stored in a Kubernetes Secret for production deployments

## Install

Development install with the bundled Postgres:

```bash
kubectl create namespace cognee

kubectl -n cognee create secret generic cognee-llm \
  --from-literal=LLM_API_KEY="$YOUR_KEY"

helm upgrade --install cognee deployment/helm \
  --namespace cognee \
  --set cognee.secrets.existingSecret=cognee-llm
```

Production-style install with external secrets and a managed database:

```bash
kubectl create namespace cognee

kubectl -n cognee create secret generic cognee-llm \
  --from-literal=LLM_API_KEY="$YOUR_KEY"

kubectl -n cognee create secret generic cognee-db \
  --from-literal=DB_USERNAME="$DB_USERNAME" \
  --from-literal=DB_PASSWORD="$DB_PASSWORD" \
  --from-literal=DB_NAME="$DB_NAME"

helm upgrade --install cognee deployment/helm \
  --namespace cognee \
  --values production-values.yaml
```

Example `production-values.yaml`:

```yaml
cognee:
  replicaCount: 2
  backendAccessControl: true
  image: "cognee/cognee:main"
  secrets:
    existingSecret: cognee-llm
  service:
    type: ClusterIP
  resources:
    requests:
      cpu: "1"
      memory: "2Gi"
    limits:
      cpu: "2"
      memory: "4Gi"
  podDisruptionBudget:
    enabled: true
    minAvailable: 1

postgres:
  enabled: false

externalDatabase:
  host: "postgres.example.internal"
  port: 5432
  existingSecret: cognee-db
  usernameKey: DB_USERNAME
  passwordKey: DB_PASSWORD
  databaseKey: DB_NAME

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: cognee.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: cognee-tls
      hosts:
        - cognee.example.com
```

## Upgrade

```bash
helm upgrade cognee deployment/helm --namespace cognee --values production-values.yaml
```

## Uninstall

```bash
helm uninstall cognee --namespace cognee
```

PVCs created by the bundled Postgres StatefulSet are not removed automatically.
Delete them only after confirming data is backed up or no longer needed.

## Local access

```bash
kubectl -n cognee port-forward svc/cognee-cognee 8000:8000
curl http://127.0.0.1:8000/health
```

## Values

| Value | Default | Description |
| --- | --- | --- |
| `nameOverride` | `""` | Override chart name labels. |
| `fullnameOverride` | `""` | Override generated resource name prefix. |
| `cognee.replicaCount` | `1` | Backend replica count when autoscaling is disabled. |
| `cognee.image` | `cognee/cognee:main` | Backend image reference. |
| `cognee.imagePullPolicy` | `IfNotPresent` | Backend image pull policy. |
| `cognee.port` | `8000` | Backend container port. |
| `cognee.service.type` | `ClusterIP` | Backend Service type. |
| `cognee.service.port` | `8000` | Backend Service port. |
| `cognee.serviceAccount.create` | `true` | Create a dedicated ServiceAccount. |
| `cognee.serviceAccount.annotations` | `{}` | ServiceAccount annotations. |
| `cognee.serviceAccount.name` | `""` | Use a specific ServiceAccount name. |
| `cognee.podSecurityContext` | non-root UID/GID `1000` | Pod-level security context. |
| `cognee.securityContext` | drops capabilities | Container-level security context. |
| `cognee.backendAccessControl` | `false` | Sets `ENABLE_BACKEND_ACCESS_CONTROL`. Use `true` for production multi-user deployments. |
| `cognee.env.HOST` | `0.0.0.0` | Backend bind host. |
| `cognee.env.ENV` | `local` | Cognee environment. |
| `cognee.env.PYTHONPATH` | `.` | Python path. |
| `cognee.env.PYTHONDONTWRITEBYTECODE` | `1` | Avoid bytecode writes on a read-only root filesystem. |
| `cognee.env.HOME` | `/tmp` | Writable home path for runtime libraries. |
| `cognee.env.DATA_ROOT_DIRECTORY` | `/var/lib/cognee/data` | Runtime data directory mounted on a writable volume. |
| `cognee.env.SYSTEM_ROOT_DIRECTORY` | `/var/lib/cognee/system` | Runtime system directory mounted on a writable volume. |
| `cognee.env.CACHE_ROOT_DIRECTORY` | `/var/lib/cognee/cache` | Runtime cache directory mounted on a writable volume. |
| `cognee.env.DB_PROVIDER` | `postgres` | Relational database provider. |
| `cognee.env.GRAPH_DATABASE_PROVIDER` | `kuzu` | Graph database provider. |
| `cognee.env.VECTOR_DB_PROVIDER` | `pgvector` | Vector database provider. |
| `cognee.env.LLM_MODEL` | `openai/gpt-4o-mini` | LLM model. |
| `cognee.env.LLM_PROVIDER` | `openai` | LLM provider. |
| `cognee.secrets.existingSecret` | `""` | Existing Secret containing the LLM API key. |
| `cognee.secrets.llmApiKeyKey` | `LLM_API_KEY` | Key name inside the LLM Secret. |
| `cognee.probes.*` | enabled | Liveness, readiness, and startup probe settings. |
| `cognee.resources.requests` | `500m`, `1Gi` | Backend CPU and memory requests. |
| `cognee.resources.limits` | `2`, `2Gi` | Backend CPU and memory limits. |
| `cognee.autoscaling.enabled` | `false` | Enable HPA for the backend Deployment. |
| `cognee.podDisruptionBudget.enabled` | `false` | Create a backend PDB. |
| `cognee.persistence.enabled` | `false` | Create a backend PVC for `/var/lib/cognee`; otherwise use `emptyDir`. |
| `cognee.persistence.existingClaim` | `""` | Existing PVC for backend runtime data. |
| `cognee.persistence.storageClassName` | `""` | StorageClass for the backend PVC. |
| `cognee.persistence.size` | `5Gi` | Backend PVC size. |
| `postgres.enabled` | `true` | Deploy bundled pgvector Postgres. Disable for managed databases. |
| `postgres.image` | `pgvector/pgvector:pg17` | Bundled Postgres image. |
| `postgres.service.port` | `5432` | Bundled Postgres Service port. |
| `postgres.auth.existingSecret` | `""` | Existing Secret for bundled Postgres credentials. |
| `postgres.auth.username` | `cognee` | Username for the generated bundled Postgres Secret. |
| `postgres.auth.database` | `cognee_db` | Database name for the generated bundled Postgres Secret. |
| `postgres.auth.*Key` | `POSTGRES_*` | Key names used for generated or existing bundled Postgres Secrets. |
| `postgres.env.PGDATA` | `/var/lib/postgresql/data/pgdata` | Postgres data directory inside the mounted PVC. |
| `postgres.storage` | `2Gi` | Bundled Postgres volume size. |
| `postgres.probes.*` | enabled | Bundled Postgres readiness/liveness probe settings. |
| `postgres.resources.requests` | `250m`, `512Mi` | Bundled Postgres CPU and memory requests. |
| `postgres.resources.limits` | `1`, `1Gi` | Bundled Postgres CPU and memory limits. |
| `externalDatabase.host` | `""` | Managed database host, required when `postgres.enabled=false`. |
| `externalDatabase.port` | `5432` | Managed database port. |
| `externalDatabase.existingSecret` | `""` | Existing Secret for managed database credentials, required when `postgres.enabled=false`. |
| `externalDatabase.usernameKey` | `DB_USERNAME` | Username key in external DB Secret. |
| `externalDatabase.passwordKey` | `DB_PASSWORD` | Password key in external DB Secret. |
| `externalDatabase.databaseKey` | `DB_NAME` | Database-name key in external DB Secret. |
| `ingress.enabled` | `false` | Create an Ingress. |
| `ingress.className` | `""` | Ingress class name. |
| `ingress.annotations` | `{}` | Ingress annotations. |
| `ingress.hosts` | `cognee.local` | Ingress hosts and paths. |
| `ingress.tls` | `[]` | Ingress TLS entries. |
| `tests.enabled` | `true` | Create a `helm test` health-check Pod. |
| `tests.image` | `curlimages/curl:8.10.1` | Image used by the health-check test hook. |

## Validation

Run these before opening a PR:

```bash
helm lint deployment/helm
helm template cognee deployment/helm --namespace cognee
helm template cognee deployment/helm --namespace cognee | kubeconform -strict -ignore-missing-schemas
helm test cognee --namespace cognee
```

If optional tools are not installed locally, run the first two commands in CI or a machine with Helm.
