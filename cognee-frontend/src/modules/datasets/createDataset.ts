import { CogneeInstance } from "../instances/types";

export default function createDataset(dataset: { name: string }, instance: CogneeInstance) {
  return instance.fetch(`/v1/datasets/`, {
    method: "POST",
    body: JSON.stringify(dataset),
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((response) => response.json());
}
