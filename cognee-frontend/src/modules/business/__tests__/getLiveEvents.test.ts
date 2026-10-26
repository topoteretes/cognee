import getLiveEvents from "../getLiveEvents";
import type { CogneeInstance } from "@/modules/instances/types";

function instance(payload: unknown): CogneeInstance {
  return {
    name: "test",
    instanceId: "tenant-x",
    fetch: () => Promise.resolve({ json: () => Promise.resolve(payload) } as Response),
  } as unknown as CogneeInstance;
}

describe("getLiveEvents — the poll validates what the socket validates", () => {
  it("keeps events that carry an identity field the dedupe key can use", async () => {
    const usable = { kind: "search", qa_id: "q1", time: "2026-01-01T00:00:00Z", question: "why" };
    const { events, cursor } = await getLiveEvents("ds-1", null, instance({ events: [usable], cursor: "c1" }));
    expect(events).toEqual([usable]);
    expect(cursor).toBe("c1");
  });

  it("drops the same shapes the socket drops, so both transports agree", async () => {
    const { events } = await getLiveEvents(
      "ds-1",
      null,
      instance({
        events: [
          { kind: "search", qa_id: "q1", time: "2026-01-01T00:00:00Z" },
          // no qa_id and no time: collapses onto every other identity-less
          // event under eventKey, so the second one silently vanishes anyway
          { kind: "search", question: "no identity field" },
          // epoch ms rather than an ISO string
          { kind: "search", qa_id: "q2", time: 1735689600000 },
          { kind: "search", qa_id: "q3", node_ids: null },
          { kind: "search", qa_id: "q4", node_ids: [{ id: "n1" }] },
          {},
        ],
        cursor: "c1",
      }),
    );
    expect(events).toEqual([{ kind: "search", qa_id: "q1", time: "2026-01-01T00:00:00Z" }]);
  });

  it("normalizes a missing cursor to null rather than passing a non-string through", async () => {
    const { cursor } = await getLiveEvents("ds-1", null, instance({ events: [], cursor: 12345 }));
    expect(cursor).toBeNull();
  });

  it("throws on a payload with no events array instead of reading as no activity", async () => {
    await expect(getLiveEvents("ds-1", null, instance({ cursor: "c1" }))).rejects.toThrow(
      "live-events response carried no events array",
    );
    await expect(getLiveEvents("ds-1", null, instance(null))).rejects.toThrow(
      "live-events response was not an object",
    );
  });
});
