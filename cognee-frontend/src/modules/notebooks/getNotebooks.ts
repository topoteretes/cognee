import { CogneeInstance } from "@/modules/instances/types";

export default function getNotebooks(instance: CogneeInstance) {
  return instance.fetch("/v1/notebooks/", {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  }).then((response) => response.json());
}
