import { fetch } from "@/utils";

export default function createDataset(dataset: { name: string }) {
  return fetch(`/v1/datasets/`, {
    method: "POST",
    body: JSON.stringify(dataset),
    headers: {
      "Content-Type": "application/json",
    }
  })
    .then((response) => response.json());
}
