import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

// The page hands each of the three choices to its own store function and its
// own control. The three store functions have the same signature, so a swap
// compiles and no other test notices. This pins which control stores into
// which function. Everything else the page renders is stubbed.
const mockAssignModel = jest.fn().mockResolvedValue(undefined);
const mockAssignPrompt = jest.fn().mockResolvedValue(undefined);
const mockAssignOntology = jest.fn().mockResolvedValue(undefined);
const mockGetData = jest.fn();
const mockPush = jest.fn();
const mockCognify = jest.fn();
const mockDeleteDataset = jest.fn();
const mockClearOutdated = jest.fn().mockResolvedValue(undefined);
const mockAddModel = jest.fn();
const mockLoadConfig = jest.fn();
const mockUpload = jest.fn();
const mockDeleteOntology = jest.fn().mockResolvedValue(undefined);
const mockRemoveOntologyAssignments = jest.fn().mockResolvedValue(undefined);
const mockDeleteCustomPrompt = jest.fn().mockResolvedValue(undefined);
const mockFlagOntologyUsers = jest.fn().mockResolvedValue(undefined);
const mockListOntologies = jest.fn();
const mockUploadOntology = jest.fn().mockResolvedValue(undefined);
let mockDatasets: { id: string; name: string; connection_id?: string }[] = [];
const mockNotify = jest.fn();
const mockFindModel = jest.fn().mockReturnValue(null);
const mockFindPrompt = jest.fn().mockReturnValue(null);
const mockFindOntology = jest.fn().mockReturnValue(null);
const mockSaveCustomPrompt = jest.fn().mockResolvedValue(undefined);
const instance = { instanceId: "workspace-1", fetch: jest.fn() };
let mockInstance: { instanceId: string } = instance;

jest.mock("next/navigation", () => ({ useRouter: () => ({ push: (...args: unknown[]) => mockPush(...args), replace: jest.fn() }) }));
jest.mock("@/modules/datasets/cognifyDataset", () => ({ __esModule: true, default: (...args: unknown[]) => mockCognify(...args) }));
jest.mock("@/modules/datasets/deleteDataset", () => ({ __esModule: true, default: (...args: unknown[]) => mockDeleteDataset(...args) }));
jest.mock("@/modules/tenant/TenantProvider", () => ({ useCogniInstance: () => ({ cogniInstance: mockInstance, isInitializing: false }) }));
jest.mock("@/ui/layout/FilterContext", () => ({ useFilter: () => ({ datasets: mockDatasets }) }));
jest.mock("@/modules/analytics", () => ({ TrackPageView: () => null, trackEvent: jest.fn() }));
jest.mock("@mantine/notifications", () => ({ notifications: { show: (...args: unknown[]) => mockNotify(...args) } }));
jest.mock("@/utils/monitoring", () => ({ captureException: jest.fn(), recordUploadSuccess: jest.fn(), recordUploadFailure: jest.fn() }));
jest.mock("@/modules/datasets/getDatasetData", () => ({
  __esModule: true,
  DEFAULT_DATASET_DATA_LIMIT: 100,
  default: (...args: unknown[]) => mockGetData(...args),
}));
// The page asks for the dataset-wide total separately from the page of rows.
jest.mock("@/modules/datasets/getDatasetDataCount", () => ({ __esModule: true, default: () => Promise.resolve(null) }));
jest.mock("@/modules/datasets/useDatasetStatuses", () => ({ useDatasetStatuses: () => ({ statusDetails: {}, refetch: jest.fn() }) }));
jest.mock("@/modules/datasets/useDatasetProcessing", () => ({ useDatasetProcessing: () => ({ data: undefined, isError: false, refetch: jest.fn() }) }));
jest.mock("@/modules/ingestion/useBrainUpload", () => ({ useBrainUpload: () => ({ stage: "idle", progress: { stage: "idle", filesTotal: 0 }, upload: (...args: unknown[]) => mockUpload(...args) }) }));
jest.mock("@/modules/billing/useLowBalanceUploadWarning", () => ({
  useLowBalanceUploadWarning: () => ({ pendingWarning: null, isEstimating: false, confirmUpload: jest.fn(), cancel: jest.fn() }),
}));
jest.mock("@/modules/ontologies/ontologyApi", () => ({
  listOntologies: (...args: unknown[]) => mockListOntologies(...args),
  uploadOntology: (...args: unknown[]) => mockUploadOntology(...args),
  deleteOntology: (...args: unknown[]) => mockDeleteOntology(...args),
}));
jest.mock("@/modules/llm/managementLlmApi", () => ({ inferSchema: jest.fn(), generateCustomPrompt: jest.fn() }));
jest.mock("@/modules/configuration/userConfiguration", () => ({
  fetchGraphModelsConfig: (...args: unknown[]) => mockLoadConfig(...args),
  datasetsAssignedTo: jest.requireActual("@/modules/configuration/userConfiguration").datasetsAssignedTo,
  withoutValue: jest.requireActual("@/modules/configuration/userConfiguration").withoutValue,
  assignGraphModelToDataset: (...args: unknown[]) => mockAssignModel(...args),
  assignPromptToDataset: (...args: unknown[]) => mockAssignPrompt(...args),
  assignOntologyToDataset: (...args: unknown[]) => mockAssignOntology(...args),
  clearDatasetOutdated: (...args: unknown[]) => mockClearOutdated(...args),
  addGraphModelForDataset: (...args: unknown[]) => mockAddModel(...args),
  findModelForDataset: (...args: unknown[]) => mockFindModel(...args),
  findPromptForDataset: (...args: unknown[]) => mockFindPrompt(...args),
  findOntologyForDataset: (...args: unknown[]) => mockFindOntology(...args),
  saveCustomPrompt: (...args: unknown[]) => mockSaveCustomPrompt(...args),
  deleteCustomPrompt: (...args: unknown[]) => mockDeleteCustomPrompt(...args),
  removeOntologyAssignments: (...args: unknown[]) => mockRemoveOntologyAssignments(...args),
  flagDatasetsUsingOntology: (...args: unknown[]) => mockFlagOntologyUsers(...args),
}));

// The customization bar is the page's three controls; the stub exposes one
// button per control and shows what each currently has selected.
jest.mock("../partials/MemoryCustomizationBar", () => ({
  __esModule: true,
  default: (props: {
    selectedModelId: string | null; onSelectModel: (id: string | null) => void;
    selectedPromptName: string | null; onSelectPrompt: (name: string | null) => void;
    selectedOntologyKey: string | null; onSelectOntology: (key: string | null) => void;
    onEditPrompt: (name: string, text: string) => void;
    onDeleteOntology: (key: string) => void;
    onCreateModel: () => void;
    onUploadOntology: () => void;
    ontologies: Record<string, unknown>;
  }) => (
    <div>
      <button onClick={() => props.onSelectModel("m1")}>pick model</button>
      <button onClick={() => props.onSelectPrompt("p1")}>pick prompt</button>
      <button onClick={() => props.onSelectOntology("onto")}>pick ontology</button>
      <button onClick={() => props.onSelectOntology(null)}>pick automatic ontology</button>
      <button onClick={() => props.onEditPrompt("p2", "new text")}>edit prompt</button>
      <button onClick={() => props.onEditPrompt("p1", "changed text")}>reword assigned prompt</button>
      <button onClick={() => props.onEditPrompt("p1", "text")}>resave assigned prompt</button>
      <button onClick={() => props.onDeleteOntology("onto")}>delete ontology</button>
      <button onClick={() => props.onCreateModel()}>create model</button>
      <button onClick={() => props.onUploadOntology()}>upload ontology</button>
      <output data-testid="ontologies">{Object.keys(props.ontologies).join(",")}</output>
      <output data-testid="selected">{`${props.selectedModelId}|${props.selectedPromptName}|${props.selectedOntologyKey}`}</output>
    </div>
  ),
}));
jest.mock("../partials/FilesTable", () => ({ __esModule: true, default: () => null }));
jest.mock("../partials/CreateModelModal", () => ({
  __esModule: true,
  default: ({ onBlank }: { onBlank: () => void }) => <button onClick={onBlank}>create blank model</button>,
}));
jest.mock("../partials/CreatePromptModal", () => ({ __esModule: true, default: () => null }));
jest.mock("../partials/PromptEditorModal", () => ({
  __esModule: true,
  default: ({ onSave, onDelete }: { onSave: () => void; onDelete: () => void }) => (
    <>
      <button onClick={onSave}>save prompt</button>
      <button onClick={onDelete}>delete prompt</button>
    </>
  ),
}));
jest.mock("../partials/UploadOntologyModal", () => ({
  __esModule: true,
  default: ({ onSubmit }: { onSubmit: (key: string, file: File) => Promise<void> }) => (
    <button onClick={() => { void onSubmit("onto", new File(["<rdf/>"], "onto.owl")).catch(() => {}); }}>submit ontology</button>
  ),
}));
jest.mock("@/ui/elements/ShareDatasetModal", () => ({ __esModule: true, default: () => null }));
jest.mock("@/ui/elements/LowBalanceWarningModal", () => ({ __esModule: true, default: () => null }));
jest.mock("@/ui/elements/DeleteConfirmModal", () => ({
  __esModule: true,
  default: ({ title, message, onConfirm }: { title: string; message: React.ReactNode; onConfirm: () => void }) => (
    <div>
      <p data-testid="confirm-message">{message}</p>
      <button onClick={onConfirm}>{`confirm: ${title}`}</button>
    </div>
  ),
}));

import DatasetDetailPage from "../DatasetDetailPage";

async function renderPage(): Promise<void> {
  render(<DatasetDetailPage datasetId="ds-1" />);
  await screen.findByText("pick model");
  // The configuration load settles on the first flush; picking before it would
  // be overwritten by it.
  await act(async () => {});
}

beforeEach(() => {
  jest.clearAllMocks();
  mockInstance = instance;
  mockDatasets = [{ id: "ds-1", name: "Contracts" }];
  mockGetData.mockResolvedValue([]);
  mockCognify.mockResolvedValue(undefined);
  mockDeleteDataset.mockResolvedValue(undefined);
  mockAddModel.mockResolvedValue(undefined);
  mockLoadConfig.mockResolvedValue({ models: [{ id: "m1", name: "Model one", assignedDatasets: [] }], customPrompts: { p1: "text" } });
  mockFindModel.mockReturnValue(null);
  mockFindPrompt.mockReturnValue(null);
  mockFindOntology.mockReturnValue(null);
  mockListOntologies.mockResolvedValue({ onto: { key: "onto" } });
});

describe("DatasetDetailPage choice wiring", () => {
  it("shows what is already stored for the dataset in each control once the configuration has loaded", async () => {
    mockFindModel.mockReturnValue({ id: "m1" });
    mockFindPrompt.mockReturnValue("p1");
    mockFindOntology.mockReturnValue("onto");

    await renderPage();

    await waitFor(() => expect(screen.getByTestId("selected")).toHaveTextContent("m1|p1|onto"));
  });

  // The page is usable before the stored configuration has been read. A pick made
  // in that window must not be replaced when the read arrives with the old value.
  it.each([
    { control: "pick model", shown: "m1|null|null", assign: mockAssignModel, choice: "m1" },
    { control: "pick prompt", shown: "null|p1|null", assign: mockAssignPrompt, choice: "p1" },
    { control: "pick ontology", shown: "null|null|onto", assign: mockAssignOntology, choice: "onto" },
  ])("keeps a choice made with $control before the configuration arrived", async ({ control, shown, assign, choice }) => {
    let arrive: (config: object) => void = () => {};
    mockLoadConfig.mockReturnValue(new Promise((done) => { arrive = done; }));
    let finishSave: () => void = () => {};
    assign.mockReturnValueOnce(new Promise<void>((done) => { finishSave = done; }));
    render(<DatasetDetailPage datasetId="ds-1" />);
    await screen.findByText(control);

    // The read lands while the pick is still being saved, then the save finishes.
    await act(async () => { fireEvent.click(screen.getByText(control)); });
    await act(async () => arrive({ models: [{ id: "m1", name: "Model one", assignedDatasets: [] }], customPrompts: { p1: "text" } }));
    expect(screen.getByTestId("selected")).toHaveTextContent(shown);
    await act(async () => finishSave());

    expect(screen.getByTestId("selected")).toHaveTextContent(shown);
    expect(assign).toHaveBeenCalledWith(instance, "ds-1", choice);
  });

  // The read of the workspace the user switched away from can arrive last.
  it("ignores a configuration read that arrives after the workspace changed", async () => {
    mockFindOntology.mockImplementation((assignments: Record<string, string>) => assignments["ds-1"] ?? null);
    let arriveFromOld: (config: object) => void = () => {};
    mockLoadConfig.mockReturnValueOnce(new Promise((done) => { arriveFromOld = done; }));
    mockLoadConfig.mockResolvedValueOnce({ models: [], customPrompts: {}, ontologyAssignments: { "ds-1": "new-onto" } });
    const { rerender } = render(<DatasetDetailPage datasetId="ds-1" />);
    await screen.findByText("pick ontology");

    mockInstance = { instanceId: "workspace-2" };
    rerender(<DatasetDetailPage datasetId="ds-1" />);
    await waitFor(() => expect(screen.getByTestId("selected")).toHaveTextContent("null|null|new-onto"));
    await act(async () => arriveFromOld({ models: [], customPrompts: {}, ontologyAssignments: { "ds-1": "old-onto" } }));

    expect(screen.getByTestId("selected")).toHaveTextContent("null|null|new-onto");
  });

  it("does not report a read of the old workspace that fails after the workspace changed", async () => {
    let failOld: (error: Error) => void = () => {};
    mockLoadConfig.mockReturnValueOnce(new Promise((_, reject) => { failOld = reject; }));
    mockLoadConfig.mockResolvedValueOnce({ models: [], customPrompts: {} });
    const { rerender } = render(<DatasetDetailPage datasetId="ds-1" />);
    await screen.findByText("pick ontology");

    mockInstance = { instanceId: "workspace-2" };
    rerender(<DatasetDetailPage datasetId="ds-1" />);
    await act(async () => {});
    await act(async () => failOld(new Error("pod unreachable")));

    expect(screen.queryByText(/could not be loaded/)).not.toBeInTheDocument();
  });

  it("ignores the old workspace's ontology list when it arrives after the workspace changed", async () => {
    let listFromOld: (list: object) => void = () => {};
    mockListOntologies.mockReturnValueOnce(new Promise((done) => { listFromOld = done; }));
    mockListOntologies.mockResolvedValueOnce({ "new-onto": { key: "new-onto" } });
    const { rerender } = render(<DatasetDetailPage datasetId="ds-1" />);
    await screen.findByText("pick ontology");

    mockInstance = { instanceId: "workspace-2" };
    rerender(<DatasetDetailPage datasetId="ds-1" />);
    await waitFor(() => expect(screen.getByTestId("ontologies")).toHaveTextContent("new-onto"));
    await act(async () => listFromOld({ "old-onto": { key: "old-onto" } }));

    expect(screen.getByTestId("ontologies")).toHaveTextContent(/^new-onto$/);
  });

  // A pick that failed is no longer showing, so the configuration that arrives
  // after it is what the control shows.
  it("shows the stored choice when a pick made before the configuration arrived could not be stored", async () => {
    mockAssignOntology.mockRejectedValueOnce(new Error("pod unreachable"));
    mockFindOntology.mockReturnValue("stored-onto");
    let arrive: (config: object) => void = () => {};
    mockLoadConfig.mockReturnValue(new Promise((done) => { arrive = done; }));
    render(<DatasetDetailPage datasetId="ds-1" />);
    await screen.findByText("pick ontology");

    await act(async () => { fireEvent.click(screen.getByText("pick ontology")); });
    await act(async () => arrive({ models: [], customPrompts: {} }));

    expect(screen.getByTestId("selected")).toHaveTextContent("null|null|stored-onto");
  });

  it("stores the graph model choice with the graph model function and shows it", async () => {
    await renderPage();

    await act(async () => { fireEvent.click(screen.getByText("pick model")); });

    expect(mockAssignModel).toHaveBeenCalledWith(instance, "ds-1", "m1");
    expect(mockAssignPrompt).not.toHaveBeenCalled();
    expect(mockAssignOntology).not.toHaveBeenCalled();
    expect(screen.getByTestId("selected")).toHaveTextContent("m1|null|null");
    // The toast names the model, not its id.
    expect(mockNotify).toHaveBeenCalledWith(expect.objectContaining({ message: '"Contracts" now uses "Model one".' }));
  });

  it("stores the prompt choice with the prompt function and shows it", async () => {
    await renderPage();

    await act(async () => { fireEvent.click(screen.getByText("pick prompt")); });

    expect(mockAssignPrompt).toHaveBeenCalledWith(instance, "ds-1", "p1");
    expect(mockAssignModel).not.toHaveBeenCalled();
    expect(mockAssignOntology).not.toHaveBeenCalled();
    expect(screen.getByTestId("selected")).toHaveTextContent("null|p1|null");
  });

  it("stores the ontology choice with the ontology function and shows it", async () => {
    await renderPage();

    await act(async () => { fireEvent.click(screen.getByText("pick ontology")); });

    expect(mockAssignOntology).toHaveBeenCalledWith(instance, "ds-1", "onto");
    expect(mockAssignModel).not.toHaveBeenCalled();
    expect(mockAssignPrompt).not.toHaveBeenCalled();
    expect(screen.getByTestId("selected")).toHaveTextContent("null|null|onto");
  });

  it("puts the choice back when it could not be stored", async () => {
    mockAssignPrompt.mockRejectedValueOnce(new Error("pod unreachable"));
    await renderPage();

    await act(async () => { fireEvent.click(screen.getByText("pick prompt")); });

    expect(screen.getByTestId("selected")).toHaveTextContent("null|null|null");
  });

  // Saving the text of a prompt does not assign it to the dataset; the page
  // selects it afterwards, which does.
  it("assigns a prompt to the dataset once its text has been saved", async () => {
    await renderPage();

    fireEvent.click(screen.getByText("edit prompt"));
    await act(async () => { fireEvent.click(await screen.findByText("save prompt")); });

    expect(mockSaveCustomPrompt).toHaveBeenCalledWith(instance, "p2", "new text");
    expect(mockAssignPrompt).toHaveBeenCalledWith(instance, "ds-1", "p2");
    expect(screen.getByTestId("selected")).toHaveTextContent("null|p2|null");
  });

  // Uploads and builds read the three choices, so none may start while one is
  // still being stored: it would run with a choice the backend does not have yet.
  describe.each([
    ["graph model", "pick model", () => mockAssignModel],
    ["prompt", "pick prompt", () => mockAssignPrompt],
    ["ontology", "pick ontology", () => mockAssignOntology],
  ])("while the %s choice is being stored", (_name, pick, assigner) => {
    async function holdSave(): Promise<() => void> {
      let finish: () => void = () => {};
      assigner().mockReturnValueOnce(new Promise<void>((done) => { finish = done; }));
      await renderPage();
      await act(async () => { fireEvent.click(screen.getByText(pick)); });
      return finish;
    }

    it("does not allow an upload from the button", async () => {
      const finish = await holdSave();
      const upload = screen.getByRole("button", { name: "Upload files" });
      expect(upload).toBeDisabled();

      await act(async () => finish());
      expect(upload).toBeEnabled();
    });

    it("refuses files chosen or dropped meanwhile, and says why", async () => {
      const finish = await holdSave();
      const input = document.querySelector<HTMLInputElement>("input[type=file]");
      expect(input).not.toBeNull();

      await act(async () => { fireEvent.change(input as HTMLInputElement, { target: { files: [new File(["x"], "a.txt", { type: "text/plain" })] } }); });

      expect(mockUpload).not.toHaveBeenCalled();
      expect(mockNotify).toHaveBeenCalledWith(expect.objectContaining({ title: "Saving your settings" }));
      await act(async () => finish());
    });
  });

  it("does not allow a sync from a connected source while a choice is being stored", async () => {
    mockDatasets = [{ id: "ds-1", name: "Contracts", connection_id: "c1" }];
    let finish: () => void = () => {};
    mockAssignPrompt.mockReturnValueOnce(new Promise<void>((done) => { finish = done; }));
    await renderPage();
    const sync = await screen.findByRole("button", { name: /^Sync$/ });
    expect(sync).toBeEnabled();

    await act(async () => { fireEvent.click(screen.getByText("pick prompt")); });
    expect(sync).toBeDisabled();

    await act(async () => finish());
    expect(sync).toBeEnabled();
  });

  // The graph was built from the old wording. The stored configuration flags the
  // dataset (see userConfiguration); this is the banner on the page.
  describe("editing the prompt this dataset uses", () => {
    async function openWithAssignedPrompt(): Promise<void> {
      mockGetData.mockResolvedValue([{ id: "f1", name: "contract.txt" }]);
      mockFindPrompt.mockReturnValue("p1");
      await renderPage();
      await waitFor(() => expect(screen.getByTestId("selected")).toHaveTextContent("null|p1|null"));
    }

    it("shows the outdated-graph banner when its wording changes", async () => {
      await openWithAssignedPrompt();
      expect(screen.queryByText(/Knowledge graph is outdated/)).not.toBeInTheDocument();

      fireEvent.click(screen.getByText("reword assigned prompt"));
      await act(async () => { fireEvent.click(await screen.findByText("save prompt")); });

      expect(await screen.findByText(/Knowledge graph is outdated/)).toBeInTheDocument();
    });

    it("shows nothing when the same wording is saved again", async () => {
      await openWithAssignedPrompt();

      fireEvent.click(screen.getByText("resave assigned prompt"));
      await act(async () => { fireEvent.click(await screen.findByText("save prompt")); });

      expect(screen.queryByText(/Knowledge graph is outdated/)).not.toBeInTheDocument();
    });
  });

  describe("with a graph that is outdated", () => {
    async function openWithOutdatedGraph(): Promise<void> {
      mockGetData.mockResolvedValue([{ id: "f1", name: "contract.txt" }]);
      mockFindPrompt.mockReturnValue("p1");
      await renderPage();
      await waitFor(() => expect(screen.getByTestId("selected")).toHaveTextContent("null|p1|null"));
      fireEvent.click(screen.getByText("reword assigned prompt"));
      await act(async () => { fireEvent.click(await screen.findByText("save prompt")); });
      await screen.findByText(/Knowledge graph is outdated/);
    }

    it("does not allow a rebuild while a choice is being stored", async () => {
      let finish: () => void = () => {};
      mockAssignOntology.mockReturnValueOnce(new Promise<void>((done) => { finish = done; }));
      await openWithOutdatedGraph();
      const rebuild = screen.getByRole("button", { name: "Rebuild graph" });
      expect(rebuild).toBeEnabled();

      await act(async () => { fireEvent.click(screen.getByText("pick ontology")); });
      expect(rebuild).toBeDisabled();

      await act(async () => finish());
      expect(rebuild).toBeEnabled();
    });

    it("names the prompt and the ontology as well as the model as reasons", async () => {
      await openWithOutdatedGraph();

      expect(screen.getByText(/graph model, prompt or ontology was changed/)).toBeInTheDocument();
    });
  });

  // An empty dataset has no graph to rebuild, so a banner about one is noise.
  it("shows no outdated banner for a dataset without files, whatever its wording does", async () => {
    mockFindPrompt.mockReturnValue("p1");
    await renderPage();
    await waitFor(() => expect(screen.getByTestId("selected")).toHaveTextContent("null|p1|null"));

    fireEvent.click(screen.getByText("reword assigned prompt"));
    await act(async () => { fireEvent.click(await screen.findByText("save prompt")); });

    expect(screen.queryByText(/Knowledge graph is outdated/)).not.toBeInTheDocument();
  });

  // The stored configuration flags a dataset outdated whether or not it has files.
  it("shows no outdated banner for a dataset without files even when the stored configuration flags it", async () => {
    mockLoadConfig.mockResolvedValue({ models: [], customPrompts: {}, outdatedDatasets: ["ds-1"] });

    await renderPage();

    expect(screen.queryByText(/Knowledge graph is outdated/)).not.toBeInTheDocument();
  });

  it("shows the outdated banner for a dataset with files that the stored configuration flags", async () => {
    mockLoadConfig.mockResolvedValue({ models: [], customPrompts: {}, outdatedDatasets: ["ds-1"] });
    mockGetData.mockResolvedValue([{ id: "f1", name: "contract.txt" }]);

    await renderPage();

    expect(await screen.findByText(/Knowledge graph is outdated/)).toBeInTheDocument();
  });

  describe("deleting what the dataset uses", () => {
    async function openWithFilesAnd(selected: { prompt?: string; ontology?: string }): Promise<void> {
      mockGetData.mockResolvedValue([{ id: "f1", name: "contract.txt" }]);
      if (selected.prompt) mockFindPrompt.mockReturnValue(selected.prompt);
      if (selected.ontology) mockFindOntology.mockReturnValue(selected.ontology);
      await renderPage();
      await waitFor(() => expect(screen.getByTestId("selected")).toHaveTextContent(`null|${selected.prompt ?? "null"}|${selected.ontology ?? "null"}`));
    }

    it("clears the ontology the dataset used, drops it from every dataset, and flags the graph", async () => {
      await openWithFilesAnd({ ontology: "onto" });

      fireEvent.click(screen.getByText("delete ontology"));
      await act(async () => { fireEvent.click(await screen.findByText("confirm: Delete ontology")); });

      expect(mockDeleteOntology).toHaveBeenCalledWith(instance, "onto");
      expect(mockRemoveOntologyAssignments).toHaveBeenCalledWith(instance, "onto");
      expect(screen.getByTestId("selected")).toHaveTextContent("null|null|null");
      expect(await screen.findByText(/Knowledge graph is outdated/)).toBeInTheDocument();
    });

    it("clears the prompt the dataset used and flags the graph", async () => {
      await openWithFilesAnd({ prompt: "p1" });

      fireEvent.click(screen.getByText("reword assigned prompt"));
      fireEvent.click(await screen.findByText("delete prompt"));
      await act(async () => { fireEvent.click(await screen.findByText("confirm: Delete prompt")); });

      expect(mockDeleteCustomPrompt).toHaveBeenCalledWith(instance, "p1");
      expect(screen.getByTestId("selected")).toHaveTextContent("null|null|null");
      expect(await screen.findByText(/Knowledge graph is outdated/)).toBeInTheDocument();
    });

    // Deleting a prompt or ontology unassigns it from every dataset that uses it.
    it("says how many other datasets a deleted prompt is taken from", async () => {
      mockLoadConfig.mockResolvedValue({ models: [], customPrompts: { p1: "text" }, promptAssignments: { "ds-1": "p1", "ds-2": "p1", "ds-3": "p1" } });
      await openWithFilesAnd({ prompt: "p1" });

      fireEvent.click(screen.getByText("reword assigned prompt"));
      fireEvent.click(await screen.findByText("delete prompt"));

      expect(await screen.findByTestId("confirm-message")).toHaveTextContent("2 other datasets use this prompt; they go back to the default prompt");
    });

    it("says nothing extra when only this dataset uses the ontology", async () => {
      mockLoadConfig.mockResolvedValue({ models: [], customPrompts: {}, ontologyAssignments: { "ds-1": "onto" } });
      await openWithFilesAnd({ ontology: "onto" });

      fireEvent.click(screen.getByText("delete ontology"));

      expect(await screen.findByTestId("confirm-message")).not.toHaveTextContent("other dataset");
    });

    // The delete unassigned it everywhere, so a later delete of the same name has no other users.
    it("stops naming other datasets once the ontology has been deleted from them", async () => {
      mockLoadConfig.mockResolvedValue({ models: [], customPrompts: {}, ontologyAssignments: { "ds-1": "onto", "ds-2": "onto" } });
      await openWithFilesAnd({ ontology: "onto" });
      fireEvent.click(screen.getByText("delete ontology"));
      await act(async () => { fireEvent.click(await screen.findByText("confirm: Delete ontology")); });

      fireEvent.click(screen.getByText("delete ontology"));

      expect(await screen.findByTestId("confirm-message")).not.toHaveTextContent("other dataset");
    });

    it("names the one other dataset that uses the ontology", async () => {
      mockLoadConfig.mockResolvedValue({ models: [], customPrompts: {}, ontologyAssignments: { "ds-1": "onto", "ds-2": "onto" } });
      await openWithFilesAnd({ ontology: "onto" });

      fireEvent.click(screen.getByText("delete ontology"));

      expect(await screen.findByTestId("confirm-message")).toHaveTextContent("1 other dataset uses this ontology");
    });

    it("leaves the selection and the banner alone when the deleted prompt was not the one in use", async () => {
      await openWithFilesAnd({ prompt: "p1" });

      fireEvent.click(screen.getByText("edit prompt"));
      fireEvent.click(await screen.findByText("delete prompt"));
      await act(async () => { fireEvent.click(await screen.findByText("confirm: Delete prompt")); });

      expect(mockDeleteCustomPrompt).toHaveBeenCalledWith(instance, "p2");
      expect(screen.getByTestId("selected")).toHaveTextContent("null|p1|null");
      expect(screen.queryByText(/Knowledge graph is outdated/)).not.toBeInTheDocument();
    });
  });

  describe("when the stored configuration cannot be read", () => {
    it("says so, and pauses uploads and builds instead of showing Automatic", async () => {
      mockLoadConfig.mockRejectedValue(new Error("pod unreachable"));

      render(<DatasetDetailPage datasetId="ds-1" />);

      expect(await screen.findByRole("alert")).toHaveTextContent("could not be loaded");
      expect(screen.getByRole("button", { name: "Upload files" })).toBeDisabled();
    });

    it("refuses files dropped meanwhile with a message that says why", async () => {
      mockLoadConfig.mockRejectedValue(new Error("pod unreachable"));
      render(<DatasetDetailPage datasetId="ds-1" />);
      await screen.findByRole("alert");
      const input = document.querySelector<HTMLInputElement>("input[type=file]");

      await act(async () => { fireEvent.change(input as HTMLInputElement, { target: { files: [new File(["x"], "a.txt", { type: "text/plain" })] } }); });

      expect(mockUpload).not.toHaveBeenCalled();
      expect(mockNotify).toHaveBeenCalledWith(expect.objectContaining({ title: "Settings not loaded" }));
    });

    it("loads them on a retry and enables uploads", async () => {
      mockLoadConfig.mockRejectedValueOnce(new Error("pod unreachable"));
      render(<DatasetDetailPage datasetId="ds-1" />);
      await screen.findByRole("alert");

      await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Try again" })); });

      await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
      expect(screen.getByRole("button", { name: "Upload files" })).toBeEnabled();
    });
  });

  it("keeps uploads paused until the configuration has been read", async () => {
    let arrive: (config: object) => void = () => {};
    mockLoadConfig.mockReturnValue(new Promise((done) => { arrive = done; }));
    render(<DatasetDetailPage datasetId="ds-1" />);
    const upload = await screen.findByRole("button", { name: "Upload files" });
    expect(upload).toBeDisabled();

    await act(async () => arrive({ models: [], customPrompts: {} }));

    expect(upload).toBeEnabled();
  });

  // A rebuild started before a choice finished building with the old choice.
  describe("clearing the outdated flag after a rebuild", () => {
    async function rebuildWithChoiceDuringBuild(): Promise<void> {
      mockGetData.mockResolvedValue([{ id: "f1", name: "contract.txt" }]);
      mockFindPrompt.mockReturnValue("p1");
      mockLoadConfig.mockResolvedValue({ models: [], customPrompts: { p1: "text" }, outdatedDatasets: ["ds-1"] });
      await renderPage();
      await screen.findByText(/Knowledge graph is outdated/);
    }

    it("clears the flag when nothing was stored during the build", async () => {
      await rebuildWithChoiceDuringBuild();

      await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Rebuild graph" })); });

      expect(mockClearOutdated).toHaveBeenCalledWith(instance, "ds-1");
    });

    it("keeps the flag when a choice was stored while the build was starting", async () => {
      await rebuildWithChoiceDuringBuild();
      let finishBuild: () => void = () => {};
      mockCognify.mockReturnValue(new Promise<void>((done) => { finishBuild = done; }));

      await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Rebuild graph" })); });
      await act(async () => { fireEvent.click(screen.getByText("pick ontology")); });
      await act(async () => finishBuild());

      expect(mockClearOutdated).not.toHaveBeenCalled();
    });

    // Rewording the prompt in use changes what the dataset builds with as much
    // as picking another one does.
    it("keeps the flag when the wording of the prompt in use was saved while the build was starting", async () => {
      await rebuildWithChoiceDuringBuild();
      let finishBuild: () => void = () => {};
      mockCognify.mockReturnValue(new Promise<void>((done) => { finishBuild = done; }));

      await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Rebuild graph" })); });
      fireEvent.click(screen.getByText("reword assigned prompt"));
      await act(async () => { fireEvent.click(await screen.findByText("save prompt")); });
      await act(async () => finishBuild());

      expect(mockSaveCustomPrompt).toHaveBeenCalledWith(instance, "p1", "changed text");
      expect(mockClearOutdated).not.toHaveBeenCalled();
    });

    it("keeps the flag when the ontology in use was deleted while the build was starting", async () => {
      mockFindOntology.mockReturnValue("onto");
      await rebuildWithChoiceDuringBuild();
      let finishBuild: () => void = () => {};
      mockCognify.mockReturnValue(new Promise<void>((done) => { finishBuild = done; }));

      await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Rebuild graph" })); });
      fireEvent.click(screen.getByText("delete ontology"));
      await act(async () => { fireEvent.click(await screen.findByText("confirm: Delete ontology")); });
      await act(async () => finishBuild());

      expect(mockRemoveOntologyAssignments).toHaveBeenCalledWith(instance, "onto");
      expect(mockClearOutdated).not.toHaveBeenCalled();
    });

    // The new model is this dataset's model now; the rebuild that started
    // before used the old one, and keeps running after the page goes to the editor.
    it("keeps the flag when a model was created for the dataset while the build was starting", async () => {
      await rebuildWithChoiceDuringBuild();
      let finishBuild: () => void = () => {};
      mockCognify.mockReturnValue(new Promise<void>((done) => { finishBuild = done; }));

      await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Rebuild graph" })); });
      fireEvent.click(screen.getByText("create model"));
      await act(async () => { fireEvent.click(await screen.findByText("create blank model")); });
      await act(async () => finishBuild());

      expect(mockAddModel).toHaveBeenCalled();
      expect(mockClearOutdated).not.toHaveBeenCalled();
    });

    it("keeps the flag when the prompt in use was deleted while the build was starting", async () => {
      await rebuildWithChoiceDuringBuild();
      let finishBuild: () => void = () => {};
      mockCognify.mockReturnValue(new Promise<void>((done) => { finishBuild = done; }));

      await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Rebuild graph" })); });
      fireEvent.click(screen.getByText("reword assigned prompt"));
      fireEvent.click(await screen.findByText("delete prompt"));
      await act(async () => { fireEvent.click(await screen.findByText("confirm: Delete prompt")); });
      await act(async () => finishBuild());

      expect(mockDeleteCustomPrompt).toHaveBeenCalledWith(instance, "p1");
      expect(mockClearOutdated).not.toHaveBeenCalled();
    });
  });

  describe("deleting an ontology the other datasets use", () => {
    it("tells the user when those datasets could not be updated", async () => {
      mockRemoveOntologyAssignments.mockRejectedValueOnce(new Error("pod unreachable"));
      mockFindOntology.mockReturnValue("onto");
      await renderPage();
      await waitFor(() => expect(screen.getByTestId("selected")).toHaveTextContent("null|null|onto"));

      fireEvent.click(screen.getByText("delete ontology"));
      await act(async () => { fireEvent.click(await screen.findByText("confirm: Delete ontology")); });

      expect(mockNotify).toHaveBeenCalledWith({
        title: "Ontology deleted, not unassigned",
        message: '"onto" is gone, but the datasets that used it could not be updated and still have it selected. Choose another ontology for each of them.',
        color: "yellow",
      });
      expect(mockNotify).not.toHaveBeenCalledWith(expect.objectContaining({ title: "Ontology deleted" }));
    });

    // This dataset is one of them: showing Automatic would hide that it still
    // builds with the deleted key, and picking Automatic would then store nothing.
    it("keeps this dataset's selection on it, and picking Automatic stores the change", async () => {
      mockRemoveOntologyAssignments.mockRejectedValueOnce(new Error("pod unreachable"));
      mockFindOntology.mockReturnValue("onto");
      await renderPage();
      await waitFor(() => expect(screen.getByTestId("selected")).toHaveTextContent("null|null|onto"));

      fireEvent.click(screen.getByText("delete ontology"));
      await act(async () => { fireEvent.click(await screen.findByText("confirm: Delete ontology")); });
      expect(screen.getByTestId("selected")).toHaveTextContent("null|null|onto");

      await act(async () => { fireEvent.click(screen.getByText("pick automatic ontology")); });
      expect(mockAssignOntology).toHaveBeenCalledWith(instance, "ds-1", null);
    });

    // Nothing about what this dataset builds with was stored.
    it("does not flag the graph when the unassign failed", async () => {
      mockRemoveOntologyAssignments.mockRejectedValueOnce(new Error("pod unreachable"));
      mockGetData.mockResolvedValue([{ id: "f1", name: "contract.txt" }]);
      mockFindOntology.mockReturnValue("onto");
      await renderPage();
      await waitFor(() => expect(screen.getByTestId("selected")).toHaveTextContent("null|null|onto"));

      fireEvent.click(screen.getByText("delete ontology"));
      await act(async () => { fireEvent.click(await screen.findByText("confirm: Delete ontology")); });

      expect(screen.queryByText(/Knowledge graph is outdated/)).not.toBeInTheDocument();
    });

    // The dataset still points at the key, so a new file under it changes what
    // it builds with although re-selecting the key stores nothing.
    it("flags the graph and the datasets using the key when an ontology is uploaded under it again", async () => {
      mockRemoveOntologyAssignments.mockRejectedValueOnce(new Error("pod unreachable"));
      mockGetData.mockResolvedValue([{ id: "f1", name: "contract.txt" }]);
      mockFindOntology.mockReturnValue("onto");
      await renderPage();
      await waitFor(() => expect(screen.getByTestId("selected")).toHaveTextContent("null|null|onto"));
      fireEvent.click(screen.getByText("delete ontology"));
      await act(async () => { fireEvent.click(await screen.findByText("confirm: Delete ontology")); });

      fireEvent.click(screen.getByText("upload ontology"));
      await act(async () => { fireEvent.click(await screen.findByText("submit ontology")); });

      expect(mockUploadOntology).toHaveBeenCalledWith(instance, "onto", expect.any(File), undefined);
      expect(mockFlagOntologyUsers).toHaveBeenCalledWith(instance, "onto");
      expect(await screen.findByText(/Knowledge graph is outdated/)).toBeInTheDocument();
    });

    it("selects and stores an uploaded ontology the dataset did not have", async () => {
      await renderPage();

      fireEvent.click(screen.getByText("upload ontology"));
      await act(async () => { fireEvent.click(await screen.findByText("submit ontology")); });

      expect(mockAssignOntology).toHaveBeenCalledWith(instance, "ds-1", "onto");
      expect(mockFlagOntologyUsers).toHaveBeenCalledWith(instance, "onto");
      expect(screen.getByTestId("selected")).toHaveTextContent("null|null|onto");
    });

    // Then nothing is known to point at the key: the failure was reading the
    // settings, and "the datasets that used it" would name datasets that do not exist.
    it("only says the settings could not be checked when no dataset is known to use the ontology", async () => {
      mockRemoveOntologyAssignments.mockRejectedValueOnce(new Error("pod unreachable"));
      await renderPage();

      fireEvent.click(screen.getByText("delete ontology"));
      await act(async () => { fireEvent.click(await screen.findByText("confirm: Delete ontology")); });

      expect(mockNotify).toHaveBeenCalledWith({
        title: "Ontology deleted",
        message: '"onto" removed, but the dataset settings could not be checked for it. If a dataset still has it selected, choose another ontology there.',
        color: "yellow",
      });
    });

    it("still names the other datasets in the next confirmation, since they were not updated", async () => {
      mockRemoveOntologyAssignments.mockRejectedValueOnce(new Error("pod unreachable"));
      mockLoadConfig.mockResolvedValue({ models: [], customPrompts: {}, ontologyAssignments: { "ds-2": "onto" } });
      await renderPage();

      fireEvent.click(screen.getByText("delete ontology"));
      await act(async () => { fireEvent.click(await screen.findByText("confirm: Delete ontology")); });
      fireEvent.click(screen.getByText("delete ontology"));

      expect(await screen.findByTestId("confirm-message")).toHaveTextContent("1 other dataset uses this ontology");
    });
  });

  // CLO-1110: these navigate after a network round trip, and a push issued after
  // the user has left lands on whatever page they went to.
  describe("a navigation that follows a request the user may have left", () => {
    it("goes to the new model once it is created", async () => {
      await renderPage();

      fireEvent.click(screen.getByText("create model"));
      await act(async () => { fireEvent.click(await screen.findByText("create blank model")); });

      expect(mockAddModel).toHaveBeenCalled();
      expect(mockPush).toHaveBeenCalledWith(expect.stringMatching(/^\/graph-models\//));
    });

    it("does not go to the new model if the user left before it was created", async () => {
      let finish: () => void = () => {};
      mockAddModel.mockReturnValue(new Promise<void>((done) => { finish = done; }));
      const { unmount } = render(<DatasetDetailPage datasetId="ds-1" />);
      await screen.findByText("create model");
      await act(async () => {});

      fireEvent.click(screen.getByText("create model"));
      await act(async () => { fireEvent.click(await screen.findByText("create blank model")); });
      unmount();
      await act(async () => finish());

      expect(mockPush).not.toHaveBeenCalled();
    });

    it("goes back to the datasets once the dataset is deleted", async () => {
      await renderPage();

      fireEvent.click(screen.getByRole("button", { name: /Delete/ }));
      await act(async () => { fireEvent.click(await screen.findByText("confirm: Delete dataset")); });

      expect(mockPush).toHaveBeenCalledWith("/datasets");
    });

    it("does not go back to the datasets if the user left before the delete finished", async () => {
      let finish: () => void = () => {};
      mockDeleteDataset.mockReturnValue(new Promise<void>((done) => { finish = done; }));
      const { unmount } = render(<DatasetDetailPage datasetId="ds-1" />);
      await screen.findByText("pick model");
      await act(async () => {});

      fireEvent.click(screen.getByRole("button", { name: /Delete/ }));
      await act(async () => { fireEvent.click(await screen.findByText("confirm: Delete dataset")); });
      unmount();
      await act(async () => finish());

      expect(mockPush).not.toHaveBeenCalled();
    });
  });
});
