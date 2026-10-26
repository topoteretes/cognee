import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useDatasetProcessing } from "@/modules/datasets/useDatasetProcessing";
import { sdkApi, type SdkConnection, type SdkSyncRun } from "@/modules/integrations/sdkApi";
import GitHubRepositories from "../GitHubRepositories";

jest.mock("@/modules/datasets/useDatasetProcessing", () => ({ useDatasetProcessing: jest.fn() }));
jest.mock("@/modules/integrations/sdkApi", () => ({ ...jest.requireActual("@/modules/integrations/sdkApi"), sdkApi: { branches: jest.fn(), selectBranch: jest.fn() } }));
const processing = jest.mocked(useDatasetProcessing);
const api = jest.mocked(sdkApi);
const onSelect = jest.fn();
const onSaved = jest.fn();
const source = "https://github.com/org/repo.git";
function show(connection: Partial<SdkConnection> = {}, branch: string | null = null) {
  return render(<GitHubRepositories resources={[{ id: "org/repo", name: "org/repo", selected: true, attributes: { branch } }]}
    selected={null} connection={{ connected: true, dataset_id: "repos", ...connection }} busy={false} onBusy={jest.fn()} onSelect={onSelect} onBranchSaved={onSaved} />);
}
function run(state: "queued" | "running" | "succeeded" | "failed", completed: number): SdkSyncRun {
  return { runId: "run", state: state === "succeeded" ? "succeeded" : "running", phase: "processing", completed: 0, total: 1, unit: "repositories", counts: {}, endedAt: null,
    resources: { [source]: { source, branch: null, state, phase: state === "running" ? "processing" : state === "succeeded" ? "complete" : state, completed, total: 3 } } };
}
beforeEach(() => {
  jest.clearAllMocks();
  processing.mockReturnValue({ data: { total: 1, completed: 1, items: [{ id: "one", name: "repo", repoUrl: source, repoBranch: null, completed: true }] }, isError: false } as unknown as ReturnType<typeof useDatasetProcessing>);
});

test("stored repository completion shows a green check and the row remains selectable", () => {
  show();
  expect(screen.getByRole("img", { name: "org/repo: Ready" })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("checkbox", { name: "org/repo" }));
  expect(onSelect).toHaveBeenCalledWith("org/repo", false);
});

test("live per-repository stages override an old completed import", () => {
  show({ sync_status: "syncing", sync_progress: [run("running", 1)] });
  expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuetext", "1 of 3 stages");
  expect(screen.queryByRole("img", { name: /Ready/ })).not.toBeInTheDocument();
});

test("a queued repository does not display stale completion", () => {
  show({ sync_status: "syncing", sync_progress: [run("queued", 0)] });
  expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "0");
});

test("the circle becomes a check only on success", () => {
  show({ sync_progress: [run("succeeded", 3)] });
  expect(screen.getByRole("img", { name: "org/repo: Ready" })).toBeInTheDocument();
});

test("failed imports show failure even if older data exists", () => {
  show({ sync_progress: [run("failed", 1)] });
  expect(screen.getByRole("img", { name: "org/repo: Failed" })).toBeInTheDocument();
});

test("changing branches never reuses another branch's completion", () => {
  show({ sync_progress: [run("succeeded", 3)] }, "dev");
  expect(screen.getByRole("progressbar", { name: "org/repo: Not imported" })).toBeInTheDocument();
});

test("branch choices come from GitHub and save the selected branch", async () => {
  api.branches.mockResolvedValue({ branches: ["main", "feature/new-ui"] });
  api.selectBranch.mockResolvedValue({ branch: "feature/new-ui" });
  show();
  fireEvent.click(screen.getByRole("button", { name: "Choose branch for org/repo" }));
  const select = await screen.findByRole("combobox", { name: "Branch for org/repo" });
  fireEvent.change(select, { target: { value: "feature/new-ui" } });
  await waitFor(() => expect(onSaved).toHaveBeenCalledWith("org/repo", "feature/new-ui"));
  expect(api.selectBranch).toHaveBeenCalledWith("org/repo", "feature/new-ui");
});

test("a failed branch save preserves the current choice and shows the error", async () => {
  api.branches.mockResolvedValue({ branches: ["main", "dev"] });
  api.selectBranch.mockRejectedValue(new Error("Branch unavailable"));
  show({}, "main");
  fireEvent.click(screen.getByRole("button", { name: "Choose branch for org/repo" }));
  fireEvent.change(await screen.findByRole("combobox"), { target: { value: "dev" } });
  expect(await screen.findByRole("alert")).toHaveTextContent("Branch unavailable");
  expect(screen.getByRole("combobox")).toHaveValue("main");
  expect(onSaved).not.toHaveBeenCalled();
});

test("a repository GitHub no longer returns offers no branch to choose", () => {
  render(<GitHubRepositories resources={[{ id: "org/gone", name: "org/gone", selected: true, unavailable: true }]}
    selected={["org/gone"]} connection={{ connected: true, dataset_id: "repos" }} busy={false} onBusy={jest.fn()} onSelect={onSelect} onBranchSaved={onSaved} />);
  expect(screen.getByRole("checkbox", { name: "org/gone" })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Choose branch for org/gone" })).not.toBeInTheDocument();
});

test("a stage the list does not know is shown as sent", () => {
  const unknown = run("running", 1);
  unknown.resources![source].phase = "uploading" as never;
  show({ sync_status: "syncing", sync_progress: [unknown] });
  expect(screen.getByRole("progressbar", { name: "org/repo: uploading" })).toBeInTheDocument();
});
