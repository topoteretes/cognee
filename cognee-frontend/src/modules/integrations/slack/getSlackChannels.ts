import { CogneeInstance } from "../../instances/types";

export interface SlackChannel {
  id: string;
  name: string;
  isPrivate: boolean;
  allowed: boolean;
}

export default function getSlackChannels(instance: CogneeInstance): Promise<{ channels: SlackChannel[] }> {
  return instance
    .fetch("/v1/slack/channels", {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    })
    .then((response) => response.json());
}
