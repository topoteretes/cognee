
# cognee-infra-helm
General infrastructure setup for Cognee on Kubernetes using a Helm chart.

## Prerequisites
Before deploying the Helm chart, ensure the following prerequisites are met: 

**Kubernetes Cluster**: A running Kubernetes cluster (e.g., Minikube, GKE, EKS).

**Helm**: Installed and configured for your Kubernetes cluster. You can install Helm by following the [official guide](https://helm.sh/docs/intro/install/). 

**kubectl**: Installed and configured to interact with your cluster. Follow the instructions [here](https://kubernetes.io/docs/tasks/tools/install-kubectl/).

Clone the Repository Clone this repository to your local machine and navigate to the directory.

## Deploy Helm Chart:

   ```bash
   helm install cognee ./cognee-chart
   ```

**Uninstall Helm Release**:
   ```bash
   helm uninstall cognee
   ```
