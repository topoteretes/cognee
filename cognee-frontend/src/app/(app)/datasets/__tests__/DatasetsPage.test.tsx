import { MantineProvider } from "@mantine/core";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import DatasetsPage from "@/app/(app)/datasets/DatasetsPage";
import { useBrainsData } from "@/app/(app)/datasets/useBrainsData";
import useDatasetFinderFilters from "@/app/(app)/datasets/partials/useDatasetFinderFilters";
import { ALL_FACET_ID, PERSONAL_FACET_ID } from "@/app/(app)/datasets/partials/datasetFacets";
import type { UseBrainsDataResult } from "@/app/(app)/datasets/brainsTypes";
import type { DatasetFinderFilters } from "@/app/(app)/datasets/partials/useDatasetFinderFilters";
import isCloudEnvironment from "@/utils/isCloudEnvironment";
import { IDLE_PROGRESS } from "@/modules/ingestion/uploadProgress";

jest.mock("@/utils/isCloudEnvironment");
jest.mock("@/app/(app)/datasets/useBrainsData");
jest.mock("@/app/(app)/datasets/partials/useDatasetFinderFilters");
jest.mock("@/modules/analytics", () => ({ TrackPageView: () => null, trackEvent: jest.fn() }));
jest.mock("@/modules/tenant/TenantProvider", () => ({ useTenant: () => ({ tenant: null }) }));
// Always mounted by DatasetsPage regardless of whether a warning is pending —
// it reads Next's router unconditionally, which this render tree doesn't provide.
jest.mock("@/ui/elements/LowBalanceWarningModal", () => () => null);

const mockUseBrainsData = jest.mocked(useBrainsData);
const mockUseDatasetFinderFilters = jest.mocked(useDatasetFinderFilters);

function brainsData(overrides: Partial<UseBrainsDataResult> = {}): UseBrainsDataResult {
  return {
    isLoading: false,
    datasets: [],
    datasetsError: false,
    sessionCreatedIds: new Set(),
    selectedId: null,
    selectedDataset: null,
    selectedDocs: [],
    docsLoading: false,
    docsError: false,
    processingCounts: undefined,
    processingCountsError: false,
    refreshProcessingCounts: jest.fn(),
    hasMoreDocs: false,
    loadMoreDocs: jest.fn(),
    retryDocs: jest.fn(),
    outdatedDatasets: new Set(),
    refreshing: false,
    isUploading: false,
    uploadStage: "idle",
    uploadProgress: { stage: "idle", filesTotal: 0, filesCompleted: 0, bytesTotal: 0, bytesSent: 0, batchesTotal: 0, batchesCompleted: 0 },
    uploadError: null,
    canRetryBuild: false,
    setUploadError: jest.fn(),
    pendingLowBalanceWarning: null,
    cancelLowBalanceUpload: jest.fn(),
    showCreateModal: true,
    setShowCreate: jest.fn(),
    newName: "new-brain",
    setNewName: jest.fn(),
    creating: false,
    createError: "",
    setCreateError: jest.fn(),
    deleteTarget: null,
    setDeleteTarget: jest.fn(),
    deleteDocTarget: null,
    setDeleteDocTarget: jest.fn(),
    deletingDocId: null,
    showPasteModal: false,
    setShowPasteModal: jest.fn(),
    pasteText: "",
    setPasteText: jest.fn(),
    pasting: false,
    handleRefresh: jest.fn(),
    handleSelectDataset: jest.fn(),
    handleUploadFiles: jest.fn(),
    handleRetryBuild: jest.fn(),
    handleDeleteFile: jest.fn(),
    handleDelete: jest.fn(),
    handleCreate: jest.fn(),
    handlePasteText: jest.fn(),
    ...overrides,
  };
}

function finderFilters(overrides: Partial<DatasetFinderFilters> = {}): DatasetFinderFilters {
  return {
    facets: [],
    facetId: ALL_FACET_ID,
    facetsStatus: "ready",
    selectFacet: jest.fn(),
    statusFilter: "all",
    setStatusFilter: jest.fn(),
    search: "",
    setSearch: jest.fn(),
    visible: [],
    narrowed: false,
    ownerLabel: () => null,
    accessFor: () => undefined,
    refreshAccess: jest.fn(),
    ...overrides,
  };
}

function renderPage(brains: UseBrainsDataResult, finder: DatasetFinderFilters) {
  mockUseBrainsData.mockReturnValue(brains);
  mockUseDatasetFinderFilters.mockReturnValue(finder);
  render(
    <MantineProvider>
      <DatasetsPage />
    </MantineProvider>,
  );
}

beforeEach(() => {
  jest.mocked(isCloudEnvironment).mockReturnValue(true);
});

afterEach(() => {
  jest.resetAllMocks();
});

describe("creating a dataset from the create modal", () => {
  it("keeps all datasets visible after creation in local mode", async () => {
    jest.mocked(isCloudEnvironment).mockReturnValue(false);
    const selectFacet = jest.fn();
    const handleCreate = jest.fn().mockResolvedValue(true);
    renderPage(brainsData({ handleCreate }), finderFilters({ selectFacet }));

    fireEvent.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() => expect(handleCreate).toHaveBeenCalledWith(null));
    expect(selectFacet).not.toHaveBeenCalled();
  });

  it("switches the rail to Personal once the dataset is actually created", async () => {
    const selectFacet = jest.fn();
    const handleCreate = jest.fn().mockResolvedValue(true);
    renderPage(brainsData({ handleCreate }), finderFilters({ selectFacet }));

    fireEvent.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() => expect(handleCreate).toHaveBeenCalledWith(null));
    expect(selectFacet).toHaveBeenCalledWith(PERSONAL_FACET_ID);
  });

  it("leaves the rail selection alone when the name is rejected and nothing was created", async () => {
    const selectFacet = jest.fn();
    const handleCreate = jest.fn().mockResolvedValue(false);
    renderPage(brainsData({ handleCreate }), finderFilters({ selectFacet }));

    fireEvent.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() => expect(handleCreate).toHaveBeenCalledWith(null));
    expect(selectFacet).not.toHaveBeenCalled();
  });
});

describe("dataset groups", () => {
  it.each([false, true])("shows the groups rail only in Cloud mode (%s)", (cloud) => {
    jest.mocked(isCloudEnvironment).mockReturnValue(cloud);
    const dataset = { id: "mine", name: "My dataset", documents: 0, status: "empty" as const };
    renderPage(
      brainsData({ datasets: [dataset], showCreateModal: false }),
      finderFilters({ visible: [dataset] }),
    );

    const groups = screen.queryByRole("navigation", { name: "Dataset groups" });
    if (cloud) expect(groups).toBeInTheDocument();
    else expect(groups).not.toBeInTheDocument();
    expect(screen.getByText("My dataset")).toBeInTheDocument();
  });
});

describe("processing progress visibility", () => {
  it.each(["estimating", "resuming", "uploading", "processing"] as const)("shows just one progress bar during %s", (stage) => {
    const dataset = { id: "mine", name: "My dataset", documents: 4, status: "running" as const };
    renderPage(
      brainsData({
        datasets: [dataset], selectedId: dataset.id, selectedDataset: dataset,
        showCreateModal: false, uploadProgress: { ...IDLE_PROGRESS, stage },
        processingCounts: { total: 4, completed: 2, pending: 2, items: [] },
      }),
      finderFilters({ visible: [dataset] }),
    );

    expect(screen.getAllByRole("progressbar")).toHaveLength(1);
    expect(screen.queryByRole("region", { name: "Processing progress" })).not.toBeInTheDocument();
  });

  it.each([
    { label: "empty dataset", total: 0, uploading: false, visible: false },
    { label: "upload before the first item arrives", total: 0, uploading: true, visible: false },
    { label: "previously imported items", total: 4, uploading: false, visible: true },
  ])("handles $label", ({ total, uploading, visible }) => {
    const dataset = { id: "mine", name: "My dataset", documents: total, status: "empty" as const };
    renderPage(
      brainsData({
        datasets: [dataset], selectedId: dataset.id, selectedDataset: dataset,
        showCreateModal: false, isUploading: uploading,
        processingCounts: { total, completed: total, pending: 0, items: [] },
      }),
      finderFilters({ visible: [dataset] }),
    );

    const progress = screen.queryByRole("region", { name: "Processing progress" });
    if (visible) expect(progress).toBeInTheDocument();
    else expect(progress).not.toBeInTheDocument();
  });
});
