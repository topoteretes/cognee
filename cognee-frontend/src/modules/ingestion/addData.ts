import { CogneeInstance } from "../instances/types";

// URL ingestion stays on /v1/add_text: the remember endpoint only accepts
// file uploads, so it would store the URL string instead of fetching it.
export async function addUrlData(dataset: { id?: string, name?: string }, url: string, instance: CogneeInstance) {
  const data = {
    textData: [url],
    datasetId: dataset.id,
    datasetName: dataset.name,
  };

  return instance.fetch("/v1/add_text", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  }).then((response) => response.json());
}
