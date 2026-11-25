import { act, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const mockPatchJson = jest.fn();
const mockGetJson = jest.fn();
jest.mock("@/services/http", () => {
  const { HttpError } = jest.requireActual("@/services/http/errors");
  return {
    HttpError,
    http: {
      patchJson: (...args: unknown[]) => mockPatchJson(...args),
      getJson: (...args: unknown[]) => mockGetJson(...args),
    },
  };
});

jest.mock("next/navigation", () => ({
  useRouter: () => ({ replace: jest.fn(), push: jest.fn() }),
  usePathname: () => "/dashboard",
}));

import { HttpError } from "@/services/http/errors";
import UserProvider from "../UserProvider";
import { useUser, type UserMe } from "../UserContext";

const ANSWERED_NEVER: UserMe = {
  isSeenWelcome: true,
  onboardingCompletedAt: "2026-01-01T00:00:00Z",
  activeWorkspaceId: "ws-1",
  pendingFeatureAnnouncements: [],
  userId: "auth0|1",
  accountCreatedAt: null,
  sessionRecordingOptIn: false,
  sessionRecordingConsentAnswered: false,
  profile: null,
};

// Reports the state the consent dialog actually gates on, and drives the
// write, so the assertions are about what a user would see.
function Probe(): React.JSX.Element {
  const { userMe, setSessionRecordingConsent } = useUser();
  return (
    <div>
      <span data-testid="answered">{String(userMe?.sessionRecordingConsentAnswered)}</span>
      <span data-testid="opted-in">{String(userMe?.sessionRecordingOptIn)}</span>
      <button onClick={() => void setSessionRecordingConsent(true, "first_login").catch(() => {})}>
        answer
      </button>
    </div>
  );
}

function renderProvider() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <UserProvider>
        <Probe />
      </UserProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  jest.clearAllMocks();
  mockGetJson.mockImplementation((url: string) =>
    url === "/api/me" ? Promise.resolve(ANSWERED_NEVER) : Promise.resolve([]),
  );
});

describe("UserProvider.setSessionRecordingConsent", () => {
  it("stops asking when the consent endpoint is not deployed", async () => {
    // The route returns 404 only for consent_endpoint_not_deployed. There is
    // no answer to store and no question worth asking, so the cache moves to
    // the unknown state — otherwise the dialog stays open on a write that
    // cannot succeed, and it has no dismissal.
    mockPatchJson.mockRejectedValue(new HttpError(404, "Not Found", "consent_endpoint_not_deployed"));
    renderProvider();
    await waitFor(() => expect(screen.getByTestId("answered").textContent).toBe("false"));

    screen.getByText("answer").click();

    await waitFor(() => expect(screen.getByTestId("answered").textContent).toBe("null"));
  });

  it("keeps asking when the write genuinely failed", async () => {
    // A 502 is a failed save, not an absent feature. Treating it the same way
    // would leave the user believing they answered while the server has no
    // record of it.
    mockPatchJson.mockRejectedValue(new HttpError(502, "Bad Gateway", "management_api_error"));
    renderProvider();
    await waitFor(() => expect(screen.getByTestId("answered").textContent).toBe("false"));

    screen.getByText("answer").click();

    await waitFor(() => expect(mockPatchJson).toHaveBeenCalled());
    expect(screen.getByTestId("answered").textContent).toBe("false");
  });

  it("writes the stored answer back into the cache on success", async () => {
    mockPatchJson.mockResolvedValue({
      ...ANSWERED_NEVER,
      sessionRecordingOptIn: true,
      sessionRecordingConsentAnswered: true,
    });
    renderProvider();
    await waitFor(() => expect(screen.getByTestId("answered").textContent).toBe("false"));

    screen.getByText("answer").click();

    await waitFor(() => expect(screen.getByTestId("answered").textContent).toBe("true"));
  });
});

describe("UserProvider cross-tab withdrawal", () => {
  const KEY = "cognee_session_recording_withdrawn";

  function withdrawalFromAnotherTab(userId: string | null): void {
    window.dispatchEvent(
      new StorageEvent("storage", { key: KEY, newValue: JSON.stringify({ userId, at: 1 }) }),
    );
  }

  it("stops recording in this tab when the same account withdraws in another", async () => {
    // Without this the tag here keeps recording until this tab's own /api/me
    // refetches — and the tab nobody is looking at is the one that waits
    // longest. Correcting the cache is what the gate reads.
    mockGetJson.mockImplementation((url: string) =>
      url === "/api/me" ? Promise.resolve({ ...ANSWERED_NEVER, sessionRecordingOptIn: true, sessionRecordingConsentAnswered: true }) : Promise.resolve([]),
    );
    renderProvider();
    await waitFor(() => expect(screen.getByTestId("opted-in").textContent).toBe("true"));

    act(() => withdrawalFromAnotherTab("auth0|1"));

    await waitFor(() => expect(screen.getByTestId("opted-in").textContent).toBe("false"));
    // No PATCH: the other tab already made the write, and repeating it here
    // would be one withdrawal stored once per open tab.
    expect(mockPatchJson).not.toHaveBeenCalled();
  });

  it("ignores a withdrawal by a different account", async () => {
    mockGetJson.mockImplementation((url: string) =>
      url === "/api/me" ? Promise.resolve({ ...ANSWERED_NEVER, sessionRecordingOptIn: true, sessionRecordingConsentAnswered: true }) : Promise.resolve([]),
    );
    renderProvider();
    await waitFor(() => expect(screen.getByTestId("opted-in").textContent).toBe("true"));

    act(() => withdrawalFromAnotherTab("auth0|2"));

    expect(screen.getByTestId("opted-in").textContent).toBe("true");
  });
});
