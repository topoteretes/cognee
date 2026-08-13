import { CogneeInstance } from "../../instances/types";

/** Mint the Slack authorize URL and return it — caller redirects the browser there. */
export default function connectSlack(instance: CogneeInstance): Promise<string> {
  return instance
    .fetch("/v1/integrations/slack/authorize", { method: "POST" })
    .then((response) => response.json())
    .then((data) => data.authorizeUrl as string);
}
