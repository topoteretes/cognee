import { CogneeInstance } from "../instances/types";

export default function getDatasets(instance: CogneeInstance) {
  return instance.fetch("/v1/datasets/", {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  }).then((response) => response.json());
}
