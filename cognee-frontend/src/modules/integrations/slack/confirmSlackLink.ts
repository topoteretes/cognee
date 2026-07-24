import { CogneeInstance } from "../../instances/types";

export default function confirmSlackLink(
  instance: CogneeInstance,
  code: string,
): Promise<{ linked: boolean }> {
  return instance
    .fetch("/v1/slack/link", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    })
    .then((response) => response.json());
}
