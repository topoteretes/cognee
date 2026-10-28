import type { ReactNode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useCodeNodeDetails } from "../useCodeNodeDetails";
import type { CogneeInstance } from "@/modules/instances/types";

function response(name: string) {
  return { json: async () => ({
    operation: "neighbors", anchor: { id: "node", kind: "module", type: "CodeModule", name },
    neighbors: [], total: 0, offset: 0, limit: 50, has_more: false,
  }) } as Response;
}

it("aborts old reads and never serves another dataset or instance's detail", async () => {
  const client = new QueryClient();
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  let finishFirst!: (response: Response) => void;
  const firstFetch = jest.fn().mockImplementation(() => new Promise<Response>((resolve) => { finishFirst = resolve; }));
  const first: CogneeInstance = { name: "test", instanceId: "one", fetch: firstFetch };
  const secondFetch = jest.fn().mockResolvedValue(response("second dataset"));
  const second: CogneeInstance = { name: "test", instanceId: "one", fetch: secondFetch };
  const { result, rerender } = renderHook(({ instance, dataset }) => useCodeNodeDetails(instance, dataset, "node"), {
    wrapper, initialProps: { instance: first, dataset: "old" },
  });
  const oldSignal = firstFetch.mock.calls[0][1].signal;
  rerender({ instance: second, dataset: "new" });
  expect(oldSignal.aborted).toBe(true);
  expect(result.current.data).toBeUndefined();
  await waitFor(() => expect(result.current.data?.pages[0].anchor.name).toBe("second dataset"));
  await act(async () => finishFirst(response("old dataset")));
  expect(result.current.data?.pages[0].anchor.name).toBe("second dataset");

  const third = { ...second, instanceId: "two", fetch: jest.fn().mockResolvedValue(response("other instance")) };
  rerender({ instance: third, dataset: "new" });
  expect(result.current.data).toBeUndefined();
  await waitFor(() => expect(result.current.data?.pages[0].anchor.name).toBe("other instance"));
});
