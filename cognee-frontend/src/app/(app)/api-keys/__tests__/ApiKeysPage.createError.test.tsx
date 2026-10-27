import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import ApiKeysPage from "../ApiKeysPage";

// The bug this pins: a failed key creation only ever reported through a
// Mantine toast, which auto-dismisses in a few seconds. A slow reader, or a
// screenshot taken later (as in CI, on a 60s Playwright timeout), saw an open
// modal with no visible reason why nothing happened — a real backend failure
// read as a silent hang (CLO-816). The fix is a persistent, testid-tagged
// error inside the modal itself, independent of the toast's lifetime.

jest.mock("@/modules/tenant/TenantProvider", () => ({
  useCogniInstance: () => ({ cogniInstance: {}, serviceUrl: "https://tenant.example.com", isInitializing: false }),
  useTenant: () => ({ tenant: { id: "t1" }, hasAccess: true, tenantReady: true }),
}));
jest.mock("@/utils", () => ({ isCloudEnvironment: () => true }));
jest.mock("@/modules/analytics", () => ({ TrackPageView: () => null, trackEvent: jest.fn() }));
jest.mock("@/modules/apiKeys/getMyUserId", () => ({ __esModule: true, default: jest.fn(async () => "user-1") }));

const mockGetApiKeys = jest.fn();
jest.mock("@/modules/apiKeys/getApiKeys", () => ({ __esModule: true, default: (...a: unknown[]) => mockGetApiKeys(...a) }));
const mockCreateApiKey = jest.fn();
jest.mock("@/modules/apiKeys/createAPIKey", () => ({ __esModule: true, default: (...a: unknown[]) => mockCreateApiKey(...a) }));
jest.mock("@/modules/apiKeys/deleteAPIKey", () => ({ __esModule: true, default: jest.fn() }));

const mockNotificationsShow = jest.fn();
jest.mock("@mantine/notifications", () => ({ notifications: { show: (...a: unknown[]) => mockNotificationsShow(...a) } }));

beforeEach(() => {
  jest.clearAllMocks();
  mockGetApiKeys.mockResolvedValue([]);
});

async function openCreateModal() {
  render(<ApiKeysPage />);
  // Two unrelated in-flight effects (loadKeys, getMyUserId) settle around here
  // and print an act() warning regardless of what this test does afterward;
  // pre-existing to the component, not something this fix introduces.
  await waitFor(() => expect(mockGetApiKeys).toHaveBeenCalled());
  fireEvent.click(screen.getAllByTestId("api-key-create-open")[0]);
  fireEvent.change(screen.getByTestId("api-key-name-input"), { target: { value: "ci-key" } });
}

describe("ApiKeysPage create-key failure", () => {
  it("shows a persistent error in the modal when the backend refuses, not only a toast", async () => {
    mockCreateApiKey.mockResolvedValue({ ok: false, error: "management API responded 500" });
    await openCreateModal();

    fireEvent.click(screen.getByTestId("api-key-create-submit"));

    const banner = await screen.findByTestId("api-key-create-error");
    expect(banner).toHaveTextContent("management API responded 500");
    // The toast fired too — real users still get it — but the banner does not
    // depend on catching it before it auto-dismisses.
    expect(mockNotificationsShow).toHaveBeenCalledWith(
      expect.objectContaining({ color: "red", message: "management API responded 500" }),
    );
  });

  it("shows a persistent error when the request throws (e.g. a client-side timeout)", async () => {
    // managementFetch's own AbortController rejects with a plain Error, not an
    // HttpError, after 30s — createApiKey rethrows it untouched, and this is
    // the path a genuinely hung backend call takes.
    mockCreateApiKey.mockRejectedValue(new Error("Request timed out."));
    await openCreateModal();

    fireEvent.click(screen.getByTestId("api-key-create-submit"));

    const banner = await screen.findByTestId("api-key-create-error");
    expect(banner).toHaveTextContent("Something went wrong while creating the key. Please try again.");
  });

  it("keeps the modal open on failure so the typed name and the error are both still visible", async () => {
    mockCreateApiKey.mockResolvedValue({ ok: false, error: "duplicate name" });
    await openCreateModal();

    fireEvent.click(screen.getByTestId("api-key-create-submit"));
    await screen.findByTestId("api-key-create-error");

    expect(screen.getByTestId("api-key-name-input")).toBeVisible();
  });

  it("clears the error on Cancel, so a retry does not start under a stale message", async () => {
    // Closing the modal unmounts its JSX either way, which would make the
    // banner disappear from the DOM even if the underlying state were never
    // reset — that would prove nothing. The real claim is about the STATE:
    // createError lives on ApiKeysPage, not inside the modal, so it survives
    // the modal closing unless something explicitly clears it. Reopening is
    // what actually distinguishes "cleared" from "just hidden".
    mockCreateApiKey.mockResolvedValue({ ok: false, error: "duplicate name" });
    await openCreateModal();
    fireEvent.click(screen.getByTestId("api-key-create-submit"));
    await screen.findByTestId("api-key-create-error");

    fireEvent.click(screen.getByText("Cancel"));
    fireEvent.click(screen.getAllByTestId("api-key-create-open")[0]);

    expect(screen.queryByTestId("api-key-create-error")).toBeNull();
  });

  it("clears the error on the backdrop click too, not only the Cancel button", async () => {
    mockCreateApiKey.mockResolvedValue({ ok: false, error: "duplicate name" });
    await openCreateModal();
    fireEvent.click(screen.getByTestId("api-key-create-submit"));
    await screen.findByTestId("api-key-create-error");

    // The overlay itself, not any element inside the panel: the panel's own
    // onClick stops propagation specifically so clicks inside it don't reach
    // this handler.
    fireEvent.click(screen.getByTestId("api-key-create-error").closest('[style*="position: fixed"]')!);
    fireEvent.click(screen.getAllByTestId("api-key-create-open")[0]);

    expect(screen.queryByTestId("api-key-create-error")).toBeNull();
  });

  it("re-enables Create after a failure, so the same click can be retried", async () => {
    mockCreateApiKey.mockResolvedValue({ ok: false, error: "duplicate name" });
    await openCreateModal();

    fireEvent.click(screen.getByTestId("api-key-create-submit"));
    await screen.findByTestId("api-key-create-error");

    expect(screen.getByTestId("api-key-create-submit")).not.toBeDisabled();
  });

  it("does not show the error banner on a successful create", async () => {
    mockCreateApiKey.mockResolvedValue({ ok: true, key: { id: "k1", key: "abc123" } });
    mockGetApiKeys.mockResolvedValue([]);
    await openCreateModal();

    fireEvent.click(screen.getByTestId("api-key-create-submit"));

    await waitFor(() => expect(mockCreateApiKey).toHaveBeenCalled());
    expect(screen.queryByTestId("api-key-create-error")).toBeNull();
  });
});
