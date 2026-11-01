import { act, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

// Signing in again after the session ended is a soft navigation: the provider
// stays mounted. These tests drive the path through /sign-in as the app does.
let mockPathname = "/dashboard";
const mockGetJson = jest.fn();
jest.mock("@/services/http", () => {
  const { HttpError } = jest.requireActual("@/services/http/errors");
  return { HttpError, http: { patchJson: jest.fn(), getJson: (...args: unknown[]) => mockGetJson(...args) } };
});
jest.mock("next/navigation", () => ({
  useRouter: () => ({ replace: jest.fn(), push: jest.fn() }),
  usePathname: () => mockPathname,
}));

import UserProvider from "../UserProvider";
import { useUser, type UserMe } from "../UserContext";

const me = (userId: string): UserMe => ({
  isSeenWelcome: true,
  onboardingCompletedAt: "2026-01-01T00:00:00Z",
  activeWorkspaceId: null,
  pendingFeatureAnnouncements: [],
  userId,
  accountCreatedAt: null,
  sessionRecordingOptIn: false,
  sessionRecordingConsentAnswered: true,
  profile: null,
});

function WhoAmI(): React.JSX.Element {
  const { userMe } = useUser();
  return <span data-testid="who">{userMe?.userId ?? "nobody"}</span>;
}

function Tree(): React.JSX.Element {
  return <UserProvider><WhoAmI /></UserProvider>;
}

let client: QueryClient;
let rerender: (ui: React.ReactElement) => void;

beforeEach(() => {
  jest.clearAllMocks();
  sessionStorage.clear();
  mockPathname = "/dashboard";
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  let account = "auth0|X";
  mockGetJson.mockImplementation((url: string) =>
    url === "/api/me" ? Promise.resolve(me(account)) : Promise.resolve([]),
  );
  (globalThis as { __signInAs?: (id: string) => void }).__signInAs = (id) => { account = id; };
  const view = render(<QueryClientProvider client={client}><Tree /></QueryClientProvider>);
  rerender = (ui) => view.rerender(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
});

function signInAs(id: string): void {
  (globalThis as unknown as { __signInAs: (id: string) => void }).__signInAs(id);
}

const goTo = (path: string) => act(async () => { mockPathname = path; rerender(<Tree />); });

describe("UserProvider across a session change in the same tab", () => {
  it("serves the new account, not the previous one, after signing in again", async () => {
    await waitFor(() => expect(screen.getByTestId("who")).toHaveTextContent("auth0|X"));
    expect(sessionStorage.getItem("cognee_session_user_id")).toBe("auth0|X");

    await goTo("/sign-in");
    signInAs("auth0|Y");
    await goTo("/");

    await waitFor(() => expect(screen.getByTestId("who")).toHaveTextContent("auth0|Y"));
    expect(sessionStorage.getItem("cognee_session_user_id")).toBe("auth0|Y");
  });

  it("wipes what the previous account cached in this tab", async () => {
    await waitFor(() => expect(screen.getByTestId("who")).toHaveTextContent("auth0|X"));
    sessionStorage.setItem("cognee_init_cache", "previous account's workspace and key");

    await goTo("/sign-in");
    signInAs("auth0|Y");
    await goTo("/");

    await waitFor(() => expect(screen.getByTestId("who")).toHaveTextContent("auth0|Y"));
    expect(sessionStorage.getItem("cognee_init_cache")).toBeNull();
  });

  // The workspace list belongs to the account too, and is fresh for a minute, so
  // without a reset the new account would be shown the previous one's list.
  it("asks for the new account's workspaces again", async () => {
    await waitFor(() => expect(screen.getByTestId("who")).toHaveTextContent("auth0|X"));
    const workspaceReads = () => mockGetJson.mock.calls.filter(([url]) => url === "/api/my-tenants").length;
    await waitFor(() => expect(workspaceReads()).toBe(1));

    await goTo("/sign-in");
    signInAs("auth0|Y");
    await goTo("/");

    await waitFor(() => expect(workspaceReads()).toBe(2));
  });

  it("does not ask for the account while on an auth page", async () => {
    await waitFor(() => expect(screen.getByTestId("who")).toHaveTextContent("auth0|X"));
    mockGetJson.mockClear();

    await goTo("/sign-in");

    expect(mockGetJson).not.toHaveBeenCalled();
    await waitFor(() => expect(screen.getByTestId("who")).toHaveTextContent("nobody"));
  });
});
