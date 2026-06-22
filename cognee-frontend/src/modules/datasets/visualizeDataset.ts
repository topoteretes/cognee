import { CogneeInstance } from "../instances/types";

export default function getVisualization(instance: CogneeInstance, datasetId: string) {
  return instance.fetch(`/v1/visualize?dataset_id=${datasetId}`, {
    method: "GET",
  }).then((response) => response);
}
