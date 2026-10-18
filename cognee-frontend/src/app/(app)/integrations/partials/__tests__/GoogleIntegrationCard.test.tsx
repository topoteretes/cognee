import { useState } from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MantineProvider } from "@mantine/core";
import { googleApi } from "@/modules/integrations/googleApi";
import { stashOAuthOutcome } from "@/modules/integrations/oauthOutcome";
import GoogleIntegrationCard from "../GoogleIntegrationCard";

jest.mock("@/modules/integrations/googleApi", () => ({
  ...jest.requireActual("@/modules/integrations/googleApi"),
  googleApi: { connection: jest.fn(), resources: jest.fn(), authorize: jest.fn(), select: jest.fn(), sync: jest.fn(), disconnect: jest.fn() },
}));
jest.mock("@/ui/elements/ModalShell", () => ({ __esModule: true, default: ({ children }: { children: React.ReactNode }) => <div role="dialog">{children}</div> }));

const api = jest.mocked(googleApi);
const connected = { connected: true, account_label: "tester@example.com", sync_status: "ok", last_synced_at: null };

beforeEach(() => {
  jest.resetAllMocks();
  window.localStorage.clear();
  window.matchMedia = jest.fn().mockReturnValue({ matches: false, addEventListener: jest.fn(), removeEventListener: jest.fn() });
  api.connection.mockResolvedValue(connected);
  api.resources.mockResolvedValue({ resources: [{ id: "a", name: "Test folder", selected: false }], selected: [] });
  api.select.mockImplementation(async (_provider, ids) => ({ selected: ids }));
  api.sync.mockResolvedValue({ accepted: true });
  api.disconnect.mockResolvedValue({ disconnected: true });
});

function Harness({ provider }: { provider: "google_drive" | "gmail" }) {
  const [connecting, setConnecting] = useState<"google_drive" | "gmail" | null>(null);
  return <GoogleIntegrationCard provider={provider} connecting={connecting} setConnecting={setConnecting} />;
}
function show(provider: "google_drive" | "gmail" = "google_drive") {
  return render(<MantineProvider><Harness provider={provider} /></MantineProvider>);
}

async function manage() {
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Manage Google Drive" }));
  await screen.findByLabelText("Test folder");
}

test("connection failure stays unknown, not disconnected, and is retryable", async () => {
  api.connection.mockRejectedValueOnce(new Error("Backend unavailable"));
  show();
  expect(await screen.findByText("Couldn't read the connection status.")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Connect Google Drive" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Retry Google Drive" }));
  expect(await screen.findByRole("button", { name: "Manage Google Drive" })).toBeEnabled();
});

test("saves scope before allowing sync and reports acceptance honestly", async () => {
  await manage();
  expect(screen.getByRole("button", { name: "Refresh" })).toBeDisabled();
  fireEvent.click(screen.getByLabelText("Test folder"));
  expect(screen.getByRole("button", { name: "Refresh" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Save selection" }));
  await screen.findByText("Selection saved.");
  expect(api.select).toHaveBeenCalledWith("google_drive", ["a"]);
  fireEvent.click(screen.getByRole("button", { name: "Refresh" }));
  expect(await screen.findByText(/Sync started/)).toBeInTheDocument();
  expect(api.sync).toHaveBeenCalledWith("google_drive");
});

test("all resources is explicit null; clearing it is an empty selection", async () => {
  await manage();
  fireEvent.click(screen.getByLabelText("Entire My Drive and all shared drives"));
  fireEvent.click(screen.getByRole("button", { name: "Save selection" }));
  await screen.findByText("Selection saved.");
  expect(api.select).toHaveBeenLastCalledWith("google_drive", null);
  fireEvent.click(screen.getByLabelText("Entire My Drive and all shared drives"));
  fireEvent.click(screen.getByRole("button", { name: "Save selection" }));
  await waitFor(() => expect(api.select).toHaveBeenLastCalledWith("google_drive", []));
});

test("does not discard selected IDs missing from the listing", async () => {
  api.resources.mockResolvedValue({ resources: [{ id: "a", name: "Test folder", selected: false }], selected: ["missing"] });
  await manage();
  expect(screen.getByLabelText("missing")).toBeChecked();
  fireEvent.click(screen.getByLabelText("Test folder"));
  fireEvent.click(screen.getByRole("button", { name: "Save selection" }));
  await waitFor(() => expect(api.select).toHaveBeenCalledWith("google_drive", ["missing", "a"]));
});

test.each([false, true])("disconnect needs confirmation and defaults to retaining data: %s", async (deleteData) => {
  await manage();
  fireEvent.click(screen.getByRole("button", { name: "Disconnect" }));
  const checkbox = screen.getByLabelText("Also delete this account’s imported Cognee dataset");
  expect(checkbox).not.toBeChecked();
  expect(api.disconnect).not.toHaveBeenCalled();
  if (deleteData) fireEvent.click(checkbox);
  fireEvent.click(screen.getByRole("button", { name: "Confirm disconnect" }));
  await waitFor(() => expect(api.disconnect).toHaveBeenCalledWith("google_drive", deleteData));
});

test("resource read failure has a retry, not an empty selection", async () => {
  api.resources.mockRejectedValueOnce(new Error("Google API unavailable"));
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Manage Google Drive" }));
  expect(await screen.findByText("Google API unavailable")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Save selection" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Retry resource list" }));
  await screen.findByLabelText("Test folder");
});

test("blocked popup does not call authorize", async () => {
  api.connection.mockResolvedValue({ connected: false });
  jest.spyOn(window, "open").mockReturnValueOnce(null);
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Connect Google Drive" }));
  fireEvent.click(screen.getByRole("button", { name: "Continue with Google Drive" }));
  expect(await screen.findByText(/Allow popups/)).toBeInTheDocument();
  expect(api.authorize).not.toHaveBeenCalled();
});

test("OAuth configuration error closes the blank popup and is visible", async () => {
  api.connection.mockResolvedValue({ connected: false });
  api.authorize.mockRejectedValue(new Error("Google integration is not configured"));
  const popup = { close: jest.fn() };
  jest.spyOn(window, "open").mockReturnValueOnce(popup as unknown as Window);
  show();
  fireEvent.click(await screen.findByRole("button", { name: "Connect Google Drive" }));
  fireEvent.click(screen.getByRole("button", { name: "Continue with Google Drive" }));
  expect(await screen.findByText(/this server has not been configured for Google sign-in/)).toBeInTheDocument();
  expect(popup.close).toHaveBeenCalled();
});

test("Gmail uses labels and shows sync failure counters", async () => {
  api.connection.mockResolvedValue({ ...connected, sync_status: "degraded", sync_counts: { skipped: 2, failed_ingestion: 1 } });
  show("gmail");
  fireEvent.click(await screen.findByRole("button", { name: "Manage Gmail" }));
  expect(await screen.findByText("Choose labels")).toBeInTheDocument();
  expect(screen.getByText("skipped: 2")).toBeInTheDocument();
  expect(screen.getByText("failed ingestion: 1")).toBeInTheDocument();
  expect(api.resources).toHaveBeenCalledWith("gmail");
});


test("connected Google card has a settled summary instead of a channel skeleton", async () => {
  const { container } = show();
  await screen.findByRole("button", { name: "Manage Google Drive" });
  expect(screen.getByText("Ready to sync")).toBeInTheDocument();
  expect(container.querySelector('[style*="skpulse"]')).toBeNull();
});

test("quota failure asks for a sync retry, not reauthorization", async () => {
  api.connection.mockResolvedValue({ ...connected, sync_status: "degraded", sync_counts: { scanned: 3, failed_rate_limit: 1 } });
  show("gmail");
  fireEvent.click(await screen.findByRole("button", { name: "Manage Gmail" }));
  expect(await screen.findByText(/Google rate limit reached/)).toBeInTheDocument();
  expect(screen.getByText(/not necessarily stored items/)).toBeInTheDocument();
  expect(screen.queryByText("Needs reconnect")).not.toBeInTheDocument();
});

test("an active sync cannot be started again", async () => {
  api.connection.mockResolvedValue({ ...connected, sync_status: "syncing" });
  api.resources.mockResolvedValue({ resources: [{ id: "a", name: "Test folder", selected: true }], selected: ["a"] });
  await manage();
  expect(screen.getByRole("button", { name: "Refresh" })).toBeDisabled();
});


test("reconnecting an already connected account reloads its resources", async () => {
  await manage();
  const popup = { closed: false, close: jest.fn(), location: { href: "" } };
  jest.spyOn(window, "open").mockReturnValueOnce(popup as unknown as Window);
  api.authorize.mockResolvedValue({ authorize_url: "https://accounts.google.com/o/oauth2/auth" });
  jest.useFakeTimers();
  try {
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Reconnect" })); });
    stashOAuthOutcome("google_drive", "connected");
    // The callback can arrive before popup.closed becomes observable.
    await act(async () => { jest.advanceTimersByTime(600); });
    expect(api.resources).toHaveBeenCalledTimes(2);
    expect(screen.getByText(/Account connected/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reconnect" })).toBeEnabled();
  } finally {
    jest.useRealTimers();
  }
});


test("active sync allows choosing folders and closing the dialog", async () => {
  api.connection.mockResolvedValue({ ...connected, sync_status: "syncing" });
  await manage();
  expect(screen.getByLabelText("Test folder")).toBeEnabled();
  fireEvent.click(screen.getByLabelText("Test folder"));
  expect(screen.getByRole("button", { name: "Save selection" })).toBeEnabled();
  fireEvent.click(screen.getByRole("button", { name: "Close" }));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

test.each(["google_drive", "gmail"] as const)("%s presents selection, sync and account actions in order", async (provider) => {
  api.connection.mockResolvedValue({ ...connected, dataset_id: "brain-1", stored_items: 164 });
  const { container } = show(provider);
  fireEvent.click(await screen.findByRole("button", { name: provider === "gmail" ? "Manage Gmail" : "Manage Google Drive" }));
  await screen.findByLabelText("Test folder");
  const sections = Array.from(container.querySelectorAll("section")).map(s => s.getAttribute("aria-label"));
  expect(sections).toEqual([provider === "gmail" ? "Choose labels" : "Choose folders", "Sync status", "Account actions"]);
  expect(screen.queryByText(/Connected means Google/)).not.toBeInTheDocument();
  expect(screen.getByRole("link", { name: "View in Brain" })).toHaveAttribute("href", "/datasets/brain-1");
  expect(screen.getByRole("button", { name: "Disconnect" })).toBeInTheDocument();
});


test("a stalled reconnect can be cancelled without trapping the dialog", async () => {
  await manage();
  const popup = { closed: false, close: jest.fn(), location: { href: "" } };
  jest.spyOn(window, "open").mockReturnValueOnce(popup as unknown as Window);
  api.authorize.mockResolvedValue({ authorize_url: "https://accounts.google.com/o/oauth2/auth" });
  fireEvent.click(screen.getByRole("button", { name: "Reconnect" }));
  await screen.findByText("Waiting for Google…");
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(popup.close).toHaveBeenCalled();
  expect(screen.getByRole("button", { name: "Reconnect" })).toBeEnabled();
  expect(screen.getByRole("button", { name: "Close" })).toBeEnabled();
});
