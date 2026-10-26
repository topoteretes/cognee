import { CogneeInstance } from "../instances/types";

export default function getDatasets(instance: CogneeInstance, signal?: AbortSignal, timeoutMs?: number) {
  const init: RequestInit & { timeoutMs?: number } = {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
    signal,
    timeoutMs,
  };
  return instance.fetch("/v1/datasets/", init).then((response) => response.json());
}
