import { CogneeInstance } from "../instances/types";

export default function deleteDataset(datasetId: string, instance: CogneeInstance) {
  return instance.fetch(`/v1/datasets/${datasetId}`, {
    method: "DELETE",
  })
}
