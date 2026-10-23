import { render, screen, waitFor } from "@testing-library/react";
import IntegrationsPage from "../../IntegrationsPage";
import { stashOAuthOutcome } from "@/modules/integrations/oauthOutcome";

jest.mock("../SdkIntegrationCard", () => ({
  __esModule: true,
  default: ({ provider }: { provider: string }) => <div data-testid="sdk-provider">{provider}</div>,
}));
jest.mock("@/modules/analytics", () => ({ TrackPageView: () => null }));
jest.mock("../SetupConnectorSection", () => ({ __esModule: true, default: () => null }));
jest.mock("../MoreDataSourcesSection", () => ({ __esModule: true, default: () => null }));
jest.mock("../useAgentConnectionStatus", () => ({ useAgentConnectionStatus: () => ({ connected: {}, lastSeenAt: {}, running: {}, loading: false }) }));
jest.mock("@/modules/integrations/oauthOutcome", () => ({ stashOAuthOutcome: jest.fn() }));

test("the integrations page includes GitHub alongside the Google SDK cards", () => {
  render(<IntegrationsPage />);
  expect(screen.getAllByTestId("sdk-provider").map((card) => card.textContent)).toEqual(["google_drive", "gmail", "github"]);
});

test("the GitHub callback relays the outcome and closes the popup", async () => {
  const opener = Object.getOwnPropertyDescriptor(window, "opener");
  Object.defineProperty(window, "opener", { configurable: true, value: {} });
  const close = jest.spyOn(window, "close").mockImplementation(() => {});
  window.history.replaceState({}, "", "/integrations?github=connected");
  try {
    render(<IntegrationsPage />);
    await waitFor(() => expect(stashOAuthOutcome).toHaveBeenCalledWith("github", "connected"));
    expect(close).toHaveBeenCalled();
  } finally {
    window.history.replaceState({}, "", "/");
    if (opener) Object.defineProperty(window, "opener", opener);
    else Reflect.deleteProperty(window, "opener");
    close.mockRestore();
  }
});
