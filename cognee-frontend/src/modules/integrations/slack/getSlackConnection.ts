import { CogneeInstance } from "../../instances/types";

export interface SlackConnection {
  connected: boolean;
  accountLabel?: string;
  providerAccountId?: string;
  connectedAt?: string;
}

export default function getSlackConnection(instance: CogneeInstance): Promise<SlackConnection> {
  return instance
    .fetch("/v1/integrations/slack/connection", {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    })
    .then((response) => response.json());
}
