const mockTrackEvent = jest.fn();
jest.mock("@/modules/analytics", () => ({
  trackEvent: (...args: unknown[]) => mockTrackEvent(...args),
}));
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useCogniInstance: () => ({ serviceUrl: "https://tenant.example", apiKey: "key-1", isInitializing: false }),
}));
jest.mock("@/ui/layout/OsPreferenceContext", () => ({
  useOsPreference: () => ({ os: "mac" }),
}));
jest.mock("next/navigation", () => ({
  useRouter: () => ({ push: jest.fn() }),
}));
jest.mock("@/app/(app)/dashboard/hooks/useDashboardTelemetry", () => ({
  useDashboardTelemetry: () => ({ sessions: [], connections: [], connectionsTruncated: false }),
}));

import { fireEvent, render, screen } from "@testing-library/react";
import type { SetupConnectorCfg } from "@/modules/integrations/types";
import SetupConnectorSection from "../SetupConnectorSection";

const N8N: SetupConnectorCfg = {
  key: "n8n",
  name: "n8n",
  cta: "Set up",
  description: "Workflows",
  icon: null,
  buildSteps: () => [
    { title: "Add the MCP node", description: "Paste this", code: "npx cognee-mcp" },
    { title: "Run it", description: "Then run" },
  ],
};

function sent(): Array<{ pageName: string; eventName: string; additionalProperties: Record<string, string> }> {
  return mockTrackEvent.mock.calls.map(([params]) => params);
}

describe("SetupConnectorSection", () => {
  beforeEach(() => {
    Object.assign(navigator, { clipboard: { writeText: jest.fn().mockResolvedValue(undefined) } });
  });

  afterEach(() => mockTrackEvent.mockReset());

  it("records the card, each step and each copy under the Integrations page and its section", () => {
    render(<SetupConnectorSection cards={[N8N]} section="automation" />);

    fireEvent.click(screen.getByRole("button", { name: /n8n/ }));
    fireEvent.click(screen.getByText("npx cognee-mcp"));
    fireEvent.click(screen.getByText("Run it"));

    expect(sent()).toEqual([
      { pageName: "Integrations", eventName: "agent_card_clicked", additionalProperties: { card: "n8n", section: "automation" } },
      {
        pageName: "Integrations",
        eventName: "agent_config_copied",
        additionalProperties: { card: "n8n", section: "automation", block: "code", step: "0" },
      },
      {
        pageName: "Integrations",
        eventName: "agent_step_viewed",
        additionalProperties: { card: "n8n", section: "automation", step: "1" },
      },
    ]);
  });
});
