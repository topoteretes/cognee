import { CogneeInstance } from "../instances/types";

export default function deleteDatasetData(datasetId: string, dataId: string, instance: CogneeInstance) {
  return instance.fetch(`/v1/datasets/${datasetId}/data/${dataId}`, {
    method: "DELETE",
  }).then((response) => {
    if (!response.ok) throw new Error(`Delete failed: ${response.status}`);
  });
}
