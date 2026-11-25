import getAgentConnectionDetail from "../getAgentConnectionDetail";
import type { CogneeInstance } from "@/modules/instances/types";

function instance(fetchImpl: jest.Mock): CogneeInstance {
  return { name: "CloudCognee", instanceId: "tenant-1", fetch: fetchImpl };
}

function jsonResponse(payload: unknown): { json: () => Promise<unknown> } {
  return { json: async () => payload };
}

describe("getAgentConnectionDetail", () => {
  it("throws on a non-object payload rather than reading it as empty lists", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(jsonResponse("not an object"));

    await expect(getAgentConnectionDetail(instance(fetchImpl), "user-1")).rejects.toThrow(
      "agents/connections/{id} returned a non-object payload",
    );
  });

  it("requests the path built from the agent's user_id", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(jsonResponse({}));

    await getAgentConnectionDetail(instance(fetchImpl), "user-1");

    expect(fetchImpl).toHaveBeenCalledWith("/v1/agents/connections/user-1", expect.objectContaining({}));
  });

  it("normalizes recent_qas, keeping question, qa_id and time", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(
      jsonResponse({
        recent_qas: [{ qa_id: "qa-1", question: "What is the release date?", answer: "Q3", time: "2026-09-29T10:00:00Z" }],
        recent_traces: [],
      }),
    );

    const detail = await getAgentConnectionDetail(instance(fetchImpl), "user-1");

    expect(detail.qas).toEqual([{ qaId: "qa-1", question: "What is the release date?", time: "2026-09-29T10:00:00Z" }]);
  });

  it("drops a qa entry missing a question, rather than rendering an empty line", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(
      jsonResponse({ recent_qas: [{ qa_id: "qa-1", time: "2026-09-29T10:00:00Z" }], recent_traces: [] }),
    );

    const detail = await getAgentConnectionDetail(instance(fetchImpl), "user-1");

    expect(detail.qas).toEqual([]);
  });

  it("drops a qa entry missing a qa_id", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(
      jsonResponse({ recent_qas: [{ question: "orphaned" }], recent_traces: [] }),
    );

    const detail = await getAgentConnectionDetail(instance(fetchImpl), "user-1");

    expect(detail.qas).toEqual([]);
  });

  it("normalizes recent_traces, preferring memory_query over origin_function as the label", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(
      jsonResponse({
        recent_qas: [],
        recent_traces: [{ trace_id: "t1", status: "ok", memory_query: "search handbook", origin_function: "search" }],
      }),
    );

    const detail = await getAgentConnectionDetail(instance(fetchImpl), "user-1");

    expect(detail.traces).toEqual([{ traceId: "t1", label: "search handbook", status: "ok", errorMessage: null, time: null }]);
  });

  it("falls back to origin_function when a trace has no memory_query", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(
      jsonResponse({ recent_qas: [], recent_traces: [{ trace_id: "t1", status: "ok", origin_function: "remember" }] }),
    );

    const detail = await getAgentConnectionDetail(instance(fetchImpl), "user-1");

    expect(detail.traces[0].label).toBe("remember");
  });

  it("keeps a trace's error_message when the pod sent one", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(
      jsonResponse({
        recent_qas: [],
        recent_traces: [{ trace_id: "t1", status: "error", origin_function: "remember", error_message: "pod timed out" }],
      }),
    );

    const detail = await getAgentConnectionDetail(instance(fetchImpl), "user-1");

    expect(detail.traces[0].errorMessage).toBe("pod timed out");
  });

  it("drops a trace with neither memory_query nor origin_function to label itself with", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(
      jsonResponse({ recent_qas: [], recent_traces: [{ trace_id: "t1", status: "ok" }] }),
    );

    const detail = await getAgentConnectionDetail(instance(fetchImpl), "user-1");

    expect(detail.traces).toEqual([]);
  });

  it("defaults qas and traces to empty lists when the pod omits them", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(jsonResponse({}));

    const detail = await getAgentConnectionDetail(instance(fetchImpl), "user-1");

    expect(detail).toEqual({ qas: [], traces: [] });
  });
});
