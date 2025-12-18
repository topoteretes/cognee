import { CogneeInstance } from "@/modules/instances/types";

export default function saveNotebook(notebookId: string, notebookData: object, instance: CogneeInstance) {
  return instance.fetch(`/v1/notebooks/${notebookId}`, {
    body: JSON.stringify(notebookData),
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
  }).then((response) => response.json());
}
