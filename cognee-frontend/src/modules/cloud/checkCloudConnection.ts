import { fetch } from "@/utils";

export default function checkCloudConnection(apiKey: string) {
  return fetch("/v1/checks/connection", {
    method: "POST",
    headers: {
      "X-Api-Key": apiKey,
    },
  });
}
