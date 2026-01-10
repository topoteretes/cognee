
# Example helm chart
Example Helm chart fro Cognee with PostgreSQL and pgvector extension
It is not ready for production usage

## Prerequisites
Before deploying the Helm chart, ensure the following prerequisites are met: 

**Kubernetes Cluster**: A running Kubernetes cluster (e.g., Minikube, GKE, EKS).

**Helm**: Installed and configured for your Kubernetes cluster. You can install Helm by following the [official guide](https://helm.sh/docs/intro/install/). 

**kubectl**: Installed and configured to interact with your cluster. Follow the instructions [here](https://kubernetes.io/docs/tasks/tools/install-kubectl/).

Clone the Repository Clone this repository to your local machine and navigate to the directory.

## Example deploy Helm Chart:

   ```bash
   helm upgrade --install cognee deployment/helm \
  --namespace cognee --create-namespace \
  --set cognee.env.LLM_API_KEY="$YOUR_KEY"
   ```

**Uninstall Helm Release**:
   ```bash
   helm uninstall cognee
   ```

## Port forwarding
To access cognee, run
```
kubectl port-forward svc/cognee-service -n cognee 8000
```
it will be available at localhost:8000
