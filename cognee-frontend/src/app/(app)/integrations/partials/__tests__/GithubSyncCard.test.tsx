import { act, fireEvent, render, screen } from "@testing-library/react";
import { githubSyncApi } from "@/modules/integrations/githubSyncApi";
import GithubSyncCard from "../GithubSyncCard";

jest.mock("@/modules/integrations/githubSyncApi", () => ({
  ...jest.requireActual("@/modules/integrations/githubSyncApi"),
  githubSyncApi: { state: jest.fn(), save: jest.fn(), runDue: jest.fn(), authorize: jest.fn(), repositories: jest.fn() },
}));
jest.mock("@/ui/elements/ModalShell", () => ({ __esModule: true, default: ({ children }: { children: React.ReactNode }) => <div role="dialog">{children}</div> }));

const api = jest.mocked(githubSyncApi);
const pushState = { settings: { enabled: true, mode: "push" as const, firstSync: "immediate" as const, repositories: { "7": { branch: "main" } } }, lastSyncedAt: null, nextSyncAt: null };

beforeEach(() => {
  jest.resetAllMocks();
  api.state.mockResolvedValue(pushState);
  api.repositories.mockResolvedValue([{ id: "7", fullName: "o/r", defaultBranch: "main" }, { id: "8", fullName: "o/other", defaultBranch: "dev" }]);
});

test("offers Connect inside the dialog when no GitHub account is connected", async () => {
  api.state.mockResolvedValue(null);
  render(<GithubSyncCard />);
  await screen.findByText("Not connected");
  fireEvent.click(screen.getByRole("button", { name: "Connect GitHub" }));
  expect(screen.getByText(/No GitHub account is connected/)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Install GitHub app" })).toBeEnabled();
});

test("a failed status read stays unknown and is retryable", async () => {
  api.state.mockRejectedValueOnce(new Error("down"));
  render(<GithubSyncCard />);
  expect(await screen.findByText("Couldn't read the connection status.")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Retry GitHub" }));
  expect(await screen.findByRole("button", { name: "Manage GitHub" })).toBeEnabled();
});

test("switching to a schedule saves it and keeps the repository filters", async () => {
  api.save.mockImplementation(async (settings) => ({ settings, syncRequested: true }));
  render(<GithubSyncCard />);
  fireEvent.click(await screen.findByRole("button", { name: "Manage GitHub" }));
  fireEvent.click(screen.getByLabelText(/On a schedule/));
  fireEvent.change(screen.getByLabelText("Minutes between syncs"), { target: { value: "30" } });
  fireEvent.click(screen.getByRole("button", { name: "Save and sync now" }));
  expect(await screen.findByText("Saved. Sync started in the background.")).toBeInTheDocument();
  const [sent, syncNow] = api.save.mock.calls[0];
  expect(sent.mode).toBe("scheduled");
  expect(sent.schedule?.intervalMinutes).toBe(30);
  expect(sent.repositories).toEqual({ "7": { branch: "main" } });
  expect(syncNow).toBe(true);
});

test("a dialog opened while the status loads fills in once it lands", async () => {
  let resolve: (value: typeof pushState) => void = () => undefined;
  api.state.mockReturnValue(new Promise((done) => { resolve = done; }));
  render(<GithubSyncCard />);
  fireEvent.click(screen.getByRole("button", { name: "Connect GitHub" }));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  resolve(pushState);
  fireEvent.click(await screen.findByRole("button", { name: "Manage GitHub" }));
  expect(await screen.findByText("Sync is running")).toBeInTheDocument();
  expect(screen.queryByText(/No GitHub account is connected/)).not.toBeInTheDocument();
});

test("adding a repository or changing a branch saves without a removal warning", async () => {
  api.save.mockImplementation(async (settings) => ({ settings, syncRequested: false }));
  render(<GithubSyncCard />);
  fireEvent.click(await screen.findByRole("button", { name: "Manage GitHub" }));
  fireEvent.click(await screen.findByLabelText("o/other"));
  fireEvent.change(screen.getByLabelText("Branch of o/other"), { target: { value: "release" } });
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await screen.findByText("Saved.");
  expect(screen.queryByRole("button", { name: "Save and remove" })).not.toBeInTheDocument();
  expect(api.save.mock.calls[0][2]).toBe(true);
  expect(api.save.mock.calls[0][0].repositories).toEqual({ "7": { branch: "main" }, "8": { branch: "release" } });
});

test("leaving a repository out names it and asks first", async () => {
  api.save.mockImplementation(async (settings) => ({ settings, syncRequested: false }));
  render(<GithubSyncCard />);
  fireEvent.click(await screen.findByRole("button", { name: "Manage GitHub" }));
  fireEvent.click(await screen.findByLabelText("o/r"));
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  expect(api.save).not.toHaveBeenCalled();
  expect(screen.getByText(/Leaving out o\/r removes its indexed data/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Save and remove" }));
  await screen.findByText("Saved.");
  expect(api.save.mock.calls[0][2]).toBe(true);
});

test("a repository checked again gets its branch back", async () => {
  render(<GithubSyncCard />);
  fireEvent.click(await screen.findByRole("button", { name: "Manage GitHub" }));
  fireEvent.click(await screen.findByLabelText("o/r"));
  fireEvent.click(screen.getByLabelText("o/r"));
  expect(screen.getByLabelText("Branch of o/r")).toHaveValue("main");
  expect(screen.queryByText(/removes/)).not.toBeInTheDocument();
});

test("typed paths are sent without leaving the field first", async () => {
  api.save.mockImplementation(async (settings) => ({ settings, syncRequested: false }));
  render(<GithubSyncCard />);
  fireEvent.click(await screen.findByRole("button", { name: "Manage GitHub" }));
  fireEvent.change(await screen.findByLabelText("Include paths of o/r"), { target: { value: "src/**" } });
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await screen.findByText("Saved.");
  expect(api.save.mock.calls[0][0].repositories?.["7"]).toEqual({ branch: "main", includePaths: ["src/**"] });
});

test("pause and sync now wait for unsaved edits and an empty interval explains itself", async () => {
  render(<GithubSyncCard />);
  fireEvent.click(await screen.findByRole("button", { name: "Manage GitHub" }));
  fireEvent.click(screen.getByLabelText(/On a schedule/));
  expect(screen.getByRole("button", { name: "Pause" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Sync now" })).toBeDisabled();
  fireEvent.change(screen.getByLabelText("Minutes between syncs"), { target: { value: "" } });
  expect(screen.getByText("Enter a whole number of minutes, 1 or more.")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
});

describe("a status read that is overtaken", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  test("a late refresh does not overwrite a newer save", async () => {
    api.save.mockImplementation(async (settings) => ({ settings, syncRequested: false }));
    render(<GithubSyncCard />);
    fireEvent.click(await screen.findByRole("button", { name: "Manage GitHub" }));
    let late: (value: typeof pushState) => void = () => undefined;
    api.state.mockReturnValue(new Promise((done) => { late = done; }));
    await act(async () => { await jest.advanceTimersByTimeAsync(10100); }); // the dialog's refresh starts
    fireEvent.click(screen.getByRole("button", { name: "Pause" }));
    expect(await screen.findByText("Sync is paused")).toBeInTheDocument();
    await act(async () => { late(pushState); }); // it answers with the state from before the pause
    expect(screen.getByText("Sync is paused")).toBeInTheDocument();
    expect(screen.queryByText(/Save your changes before pausing/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Resume" })).toBeEnabled();
  });
});

describe("installing the GitHub app", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  async function install() {
    api.state.mockResolvedValue(null);
    api.authorize.mockResolvedValue("https://github.com/apps/x/installations/new");
    const popup = { location: { href: "" }, close: jest.fn(), closed: false };
    jest.spyOn(window, "open").mockReturnValue(popup as unknown as Window);
    render(<GithubSyncCard />);
    await screen.findByText("Not connected");
    fireEvent.click(screen.getByRole("button", { name: "Connect GitHub" }));
    fireEvent.click(screen.getByRole("button", { name: "Install GitHub app" }));
    await act(async () => { await jest.advanceTimersByTimeAsync(0); });
    return popup;
  }

  test("a connection that appears closes the popup and shows the settings", async () => {
    const popup = await install();
    expect(popup.location.href).toContain("github.com");
    api.state.mockResolvedValue(pushState);
    await act(async () => { await jest.advanceTimersByTimeAsync(1600); });
    expect(popup.close).toHaveBeenCalled();
    expect(await screen.findByText("Sync is running")).toBeInTheDocument();
  });

  test("giving up after the timeout says so", async () => {
    await install();
    await act(async () => { await jest.advanceTimersByTimeAsync(305000); });
    expect(screen.getByRole("alert")).toHaveTextContent("did not finish the install in time");
  });

  test("a server error stops the wait instead of polling on", async () => {
    await install();
    api.state.mockRejectedValue(new Error("Backend unavailable"));
    await act(async () => { await jest.advanceTimersByTimeAsync(1600); });
    expect(screen.getByRole("alert")).toHaveTextContent("Backend unavailable");
    const calls = api.state.mock.calls.length;
    await act(async () => { await jest.advanceTimersByTimeAsync(10000); });
    expect(api.state.mock.calls.length).toBe(calls);
  });

  test("a slow status read is not stacked with another while the install waits", async () => {
    await install();
    let slow: (value: typeof pushState | null) => void = () => undefined;
    api.state.mockReset();
    api.state.mockReturnValue(new Promise((done) => { slow = done; }));
    await act(async () => { await jest.advanceTimersByTimeAsync(6000); });
    expect(api.state).toHaveBeenCalledTimes(1);
    await act(async () => { slow(pushState); });
    expect(await screen.findByText("Sync is running")).toBeInTheDocument();
  });
});
