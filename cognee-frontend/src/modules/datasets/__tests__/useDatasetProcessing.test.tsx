import type { ReactNode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useDatasetProcessing } from "../useDatasetProcessing";

const mockFetch = jest.fn();
const mockInstance = { fetch: mockFetch };
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useCogniInstance: () => ({ cogniInstance: mockInstance, isInitializing: false }),
  useTenant: () => ({ tenant: { tenant_id: "tenant" }, tenantReady: true }),
}));

const counts = (completed: number) => ({ total: 3, completed, pending: 3 - completed, items: [] });
const response = (body: unknown) => ({ ok: true, json: async () => body });
beforeEach(() => mockFetch.mockReset());
function wrapper() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  };
}

test("loads persisted completion after navigation and never shows another brain's counts", async () => {
  mockFetch.mockResolvedValueOnce(response(counts(2)));
  const { result, rerender } = renderHook(({ id }) => useDatasetProcessing(id), { initialProps: { id: "drive" }, wrapper: wrapper() });
  await waitFor(() => expect(result.current.data?.completed).toBe(2));
  let finish!: (value: unknown) => void;
  mockFetch.mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
  rerender({ id: "gmail" });
  expect(result.current.data).toBeUndefined();
  await act(async () => { finish(response(counts(1))); });
  await waitFor(() => expect(result.current.data?.completed).toBe(1));
  expect(mockFetch).toHaveBeenLastCalledWith("/v1/datasets/gmail/processing-status", expect.objectContaining({ signal: expect.any(AbortSignal) }));
});

test("keeps the last counts but flags them stale if refresh fails", async () => {
  mockFetch.mockResolvedValueOnce(response(counts(2)));
  const { result } = renderHook(() => useDatasetProcessing("drive"), { wrapper: wrapper() });
  await waitFor(() => expect(result.current.data?.completed).toBe(2));
  expect(result.current.isError).toBe(false);
  mockFetch.mockResolvedValueOnce({ ok: false });
  await act(async () => { await result.current.refetch(); });
  await waitFor(() => expect(result.current.isError).toBe(true));
  expect(result.current.data?.completed).toBe(2);
});
