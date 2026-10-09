import type { SetupConnectorCfg } from "./types";
import { imgIcon } from "./connectorHelpers";

export const AUTOMATION_CARDS: SetupConnectorCfg[] = [
  {
    key: "n8n",
    name: "n8n",
    cta: "Connect via node",
    description: "Add Cognee memory to your automation workflows.",
    icon: imgIcon("/visuals/logos/n8n.svg", "n8n"),
    buildSteps: (baseUrl, apiKey, loading) => [
      { title: "Install the Cognee community node", description: "In n8n, open Settings → Community Nodes → Install and enter the package name.", code: "n8n-nodes-cognee", codeToCopy: "n8n-nodes-cognee" },
      { title: "Create the Cognee API credential", description: "Add a \"Cognee API\" credential, paste your Base URL and API key, and click Test — it verifies the connection against your tenant.", code: baseUrl, codeToCopy: `${baseUrl}\n${apiKey}`, loading },
      { title: "Test the connection", description: "Add a Cognee node to a workflow (or attach it to an AI Agent) and run it — it should answer from your brain." },
    ],
  },
  {
    key: "dify",
    name: "Dify",
    cta: "Connect via plugin",
    description: "Add Cognee tools to your Dify agents and workflows.",
    icon: imgIcon("/visuals/logos/dify.svg", "Dify"),
    buildSteps: (baseUrl, apiKey, loading) => [
      { title: "Install the Cognee plugin", description: "In your Dify workspace, open the Marketplace, search for \"Cognee\" (by topoteretes), and click Install.", code: "marketplace.dify.ai/plugin/topoteretes/cognee", codeToCopy: "https://marketplace.dify.ai/plugin/topoteretes/cognee" },
      // The Dify plugin needs the /api suffix — without it, validation passes (root /health) but every tool call 404s.
      { title: "Configure the plugin", description: "Open the plugin's authorization settings and enter your Cognee Base URL (including the /api suffix) and API key.", code: `${baseUrl}/api`, codeToCopy: `${baseUrl}/api\n${apiKey}`, loading },
      { title: "Add Cognee tools to your app", description: "In an Agent or Workflow app, add the Cognee tools — create a dataset, ingest text or files, run Cognify, then search your memory." },
      { title: "Test the connection", description: "Run the app and ask: \"What do you know from cognee?\" — the Cognee search tool should answer from your brain." },
    ],
  },
];
