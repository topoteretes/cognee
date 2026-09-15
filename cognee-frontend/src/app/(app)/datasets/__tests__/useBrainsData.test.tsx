import { act, renderHook, waitFor } from "@testing-library/react";
import { useBrainsData } from "../useBrainsData";
import deleteDatasetData from "@/modules/datasets/deleteDatasetData";

const mockFetch = jest.fn();
const mockInstance = { name: "test", instanceId: "test", fetch: mockFetch };
const mockDatasets = [{ id: "a", name: "A" }, { id: "b", name: "B" }];
const mockStatuses = {};
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useCogniInstance: () => ({ cogniInstance: mockInstance, isInitializing: false }),
  useTenant: () => ({ tenant: null }),
}));
jest.mock("@/ui/layout/FilterContext", () => ({
  useFilter: () => ({ datasets: mockDatasets, refreshDatasets: jest.fn() }),
}));
jest.mock("@/modules/datasets/getDatasets", () => ({ __esModule: true, default: async () => mockDatasets }));
jest.mock("@/modules/datasets/deleteDatasetData", () => ({ __esModule: true, default: jest.fn() }));
jest.mock("@/modules/datasets/useDatasetStatuses", () => ({ useDatasetStatuses: () => ({ statusDetails: mockStatuses }) }));
jest.mock("@/modules/configuration/userConfiguration", () => ({ loadGraphModelsConfig: async () => ({}) }));
jest.mock("@/modules/ingestion/useBrainUpload", () => ({ useBrainUpload: () => ({}) }));
jest.mock("@/modules/billing/useLowBalanceUploadWarning", () => ({ useLowBalanceUploadWarning: () => ({}) }));
jest.mock("@/modules/analytics", () => ({ trackEvent: jest.fn() }));
jest.mock("@/utils/monitoring", () => ({ captureException: jest.fn() }));
const response = (body: unknown) => ({ ok: true, json: async () => body }) as Response;

test("finishing a deletion in A does not abort the selected dataset B request", async () => {
  let resolveDelete!: () => void;
  let resolveB!: (response: Response) => void;
  let bSignal: AbortSignal | undefined;
  jest.mocked(deleteDatasetData).mockImplementation(() => new Promise<void>(resolve => { resolveDelete = resolve; }));
  mockFetch.mockImplementation((url: string, options?: { signal?: AbortSignal }) => {
    if (url.endsWith("/count")) return Promise.resolve(response({ count: 1 }));
    if (url.includes("/a/data?")) return Promise.resolve(response([{ id: "a-file", name: "A" }]));
    if (url.includes("/b/data?")) {
      bSignal = options?.signal;
      return new Promise<Response>(resolve => { resolveB = resolve; });
    }
    return Promise.resolve(response({}));
  });
  const { result } = renderHook(useBrainsData);
  await waitFor(() => expect(result.current.isLoading).toBe(false));
  await act(async () => { await result.current.handleSelectDataset("a"); });
  let deleting!: Promise<void>;
  act(() => { deleting = result.current.handleDeleteFile("a-file"); });
  let selecting!: Promise<void>;
  act(() => { selecting = result.current.handleSelectDataset("b"); });
  await act(async () => { resolveDelete(); await deleting; });
  expect(bSignal?.aborted).toBe(false);
  await act(async () => { resolveB(response([{ id: "b-file", name: "B" }])); await selecting; });
  expect(result.current.selectedDocs).toEqual([{ id: "b-file", name: "B" }]);
});
