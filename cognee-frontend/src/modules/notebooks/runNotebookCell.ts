import { Cell } from "@/ui/elements/Notebook/types";
import { CogneeInstance } from "@/modules/instances/types";

export default function runNotebookCell(notebookId: string, cell: Cell, instance: CogneeInstance) {
  return instance.fetch(`/v1/notebooks/${notebookId}/${cell.id}/run`, {
    body: JSON.stringify({
      content: cell.content,
    }),
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  }).then((response: Response) => response.json());
}
