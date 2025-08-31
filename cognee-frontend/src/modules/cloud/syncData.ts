import { fetch } from "@/utils";

export default function syncData(datasetId: string) {
  return fetch("/v1/sync", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ datasetId }),
  });
}
