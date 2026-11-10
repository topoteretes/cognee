import { renderHook, waitFor } from "@testing-library/react";
import { useAgentConnectionDetail } from "../useAgentConnectionDetail";
import type { CogneeInstance } from "@/modules/instances/types";

function instance(fetchImpl: jest.Mock): CogneeInstance {
  return { name: "CloudCognee", instanceId: "tenant-1", fetch: fetchImpl };
}

function jsonResponse(payload: unknown): { json: () => Promise<unknown> } {
  return { json: async () => payload };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe("useAgentConnectionDetail", () => {
  it("is idle when agentId is null, making no fetch", () => {
    const fetchImpl = jest.fn();
    const cogniInstance = instance(fetchImpl);
    const { result } = renderHook(() => useAgentConnectionDetail(cogniInstance, null));

    expect(result.current).toEqual({ qas: [], traces: [], loading: false, failed: false });
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("fetches once when opened with an agentId, reporting loading then the result", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(
      jsonResponse({ recent_qas: [{ qa_id: "qa-1", question: "q", time: null }], recent_traces: [] }),
    );
    const cogniInstance = instance(fetchImpl);
    const { result } = renderHook(() => useAgentConnectionDetail(cogniInstance, "user-1"));

    expect(result.current.loading).toBe(true);

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.qas).toEqual([{ qaId: "qa-1", question: "q", time: null }]);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("does not refetch on rerender with the same instance and agentId", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(jsonResponse({}));
    const cogniInstance = instance(fetchImpl);
    const { result, rerender } = renderHook(
      ({ agentId }: { agentId: string | null }) => useAgentConnectionDetail(cogniInstance, agentId),
      { initialProps: { agentId: "user-1" } },
    );

    await waitFor(() => expect(result.current.loading).toBe(false));
    rerender({ agentId: "user-1" });

    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("reports failed, without throwing, on a rejected fetch", async () => {
    const fetchImpl = jest.fn().mockRejectedValue(new Error("network down"));
    const cogniInstance = instance(fetchImpl);
    const { result } = renderHook(() => useAgentConnectionDetail(cogniInstance, "user-1"));

    await waitFor(() => expect(result.current.failed).toBe(true));

    expect(result.current.qas).toEqual([]);
    expect(result.current.loading).toBe(false);
  });

  it("aborts the in-flight fetch on unmount", () => {
    const fetchImpl = jest.fn().mockReturnValue(deferred().promise);
    const cogniInstance = instance(fetchImpl);
    const { unmount } = renderHook(() => useAgentConnectionDetail(cogniInstance, "user-1"));

    const signal = (fetchImpl.mock.calls[0][1] as { signal: AbortSignal }).signal;
    expect(signal.aborted).toBe(false);

    unmount();

    expect(signal.aborted).toBe(true);
  });

  it("ignores a response that lands after agentId changes, never mixing one agent's memory into another's state", async () => {
    const first = deferred<{ json: () => Promise<unknown> }>();
    const second = deferred<{ json: () => Promise<unknown> }>();
    const fetchImpl = jest.fn().mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const cogniInstance = instance(fetchImpl);

    const { result, rerender } = renderHook(
      ({ agentId }: { agentId: string }) => useAgentConnectionDetail(cogniInstance, agentId),
      { initialProps: { agentId: "user-1" } },
    );

    rerender({ agentId: "user-2" });
    second.resolve(jsonResponse({ recent_qas: [{ qa_id: "qa-2", question: "second agent", time: null }], recent_traces: [] }));
    await waitFor(() => expect(result.current.loading).toBe(false));

    // The stale first-agent response arrives late; it must not overwrite
    // the second agent's already-landed result. A macrotask flush (not just
    // one or two microtask turns) so every .then hop in the fetch chain
    // (instance.fetch → .json() → the hook's own .then) has a chance to run.
    first.resolve(jsonResponse({ recent_qas: [{ qa_id: "qa-1", question: "first agent", time: null }], recent_traces: [] }));
    await waitFor(() => expect(fetchImpl).toHaveBeenCalledTimes(2));
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(result.current.qas).toEqual([{ qaId: "qa-2", question: "second agent", time: null }]);
  });
});
