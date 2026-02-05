import { CogneeInstance } from "@/modules/instances/types";

export default function deleteNotebook(notebookId: string, instance: CogneeInstance) {
  return instance.fetch(`/v1/notebooks/${notebookId}`, {
    method: "DELETE",
  });
}
