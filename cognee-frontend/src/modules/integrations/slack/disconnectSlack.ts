import { CogneeInstance } from "../../instances/types";

export default function disconnectSlack(instance: CogneeInstance): Promise<{ disconnected: boolean }> {
  return instance
    .fetch("/v1/integrations/slack/connection", { method: "DELETE" })
    .then((response) => response.json());
}
