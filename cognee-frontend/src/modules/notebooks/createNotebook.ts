import { CogneeInstance } from "@/modules/instances/types";

export default function createNotebook(notebookName: string, instance: CogneeInstance) {
  return instance.fetch("/v1/notebooks/", {
    body: JSON.stringify({ name: notebookName }),
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  }).then((response: Response) => response.json());
}
