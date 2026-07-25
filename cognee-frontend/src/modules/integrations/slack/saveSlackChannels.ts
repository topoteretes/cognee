import { CogneeInstance } from "../../instances/types";

export default function saveSlackChannels(
  instance: CogneeInstance,
  channelIds: string[],
): Promise<{ allowedChannelIds: string[] }> {
  return instance
    .fetch("/v1/slack/channels", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ channelIds }),
    })
    .then((response) => response.json());
}
