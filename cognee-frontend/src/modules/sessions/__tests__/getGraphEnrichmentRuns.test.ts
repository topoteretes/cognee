import { getGraphEnrichmentRuns } from "../getSessions";

import type { CogneeInstance } from "@/modules/instances/types";

const DATASET_ID = "ds-1";

interface RawRun {
  id: string;
  pipeline_name?: string;
  status: string;
  dataset_id?: string;
  created_at: string;
  pipeline_run_id: string;
  error?: string;
  error_class?: string;
  session_ids?: string[];
  session_id?: string | null;
}

function makeRun(overrides: Partial<RawRun> & Pick<RawRun, "id" | "status" | "created_at">): RawRun {
  return {
    pipeline_name: "memify_pipeline",
    dataset_id: DATASET_ID,
    pipeline_run_id: overrides.id,
    ...overrides,
  };
}

function instanceReturning(rows: RawRun[]): CogneeInstance {
  return {
    name: "test",
    instanceId: "test",
    fetch: jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(rows),
    }) as unknown as typeof global.fetch,
  };
}

/** The URL the fetcher asked the pod for. */
function requestedUrl(instance: CogneeInstance): string {
  return String((instance.fetch as unknown as jest.Mock).mock.calls[0][0]);
}

describe("getGraphEnrichmentRuns session scope (CLO-1105)", () => {
  const SESSION = "claude_abc";
  const OTHER = "claude_other";

  it("asks the pod for the session's rows and keeps only runs that list it", async () => {
    // A pod running cognee PR 6191 lists every session a run belongs to and
    // filters on `session_id`; the client-side filter is belt and braces for
    // a pod that stamps but does not filter.
    const instance = instanceReturning([
      makeRun({ id: "mine", status: "DATASET_PROCESSING_COMPLETED", created_at: "2026-07-17T10:00:00", session_ids: [SESSION] }),
      makeRun({ id: "both", status: "DATASET_PROCESSING_COMPLETED", created_at: "2026-07-17T10:00:10", session_ids: [OTHER, SESSION] }),
      makeRun({ id: "theirs", status: "DATASET_PROCESSING_ERRORED", created_at: "2026-07-17T10:00:20", session_ids: [OTHER] }),
    ]);

    const runs = await getGraphEnrichmentRuns(instance, DATASET_ID, SESSION);

    expect(requestedUrl(instance)).toContain(`session_id=${SESSION}`);
    expect(runs).toHaveLength(1);
    expect(runs[0].count).toBe(2);
    // The other session's failure must not reach this session's card.
    expect(runs[0].status).toBe("completed");
    expect(runs[0].error_count).toBe(0);
  });

  it("matches the single session_id a pod running only cognee PR 6160 sends", async () => {
    const instance = instanceReturning([
      makeRun({ id: "mine", status: "DATASET_PROCESSING_COMPLETED", created_at: "2026-07-17T10:00:00", session_id: SESSION }),
      makeRun({ id: "theirs", status: "DATASET_PROCESSING_ERRORED", created_at: "2026-07-17T10:00:20", session_id: OTHER }),
    ]);

    const runs = await getGraphEnrichmentRuns(instance, DATASET_ID, SESSION);

    expect(runs).toHaveLength(1);
    expect(runs[0].status).toBe("completed");
  });

  it("keeps the dataset-wide view on a pod that stamps no session, rather than showing nothing", async () => {
    // A pod from before the stamp ignores the query param and names no
    // session on any row. That is the card as it was before CLO-1105, and
    // an empty card there would claim the session was never bridged.
    const instance = instanceReturning([
      makeRun({ id: "a", status: "DATASET_PROCESSING_COMPLETED", created_at: "2026-07-17T10:00:00" }),
      makeRun({ id: "b", status: "DATASET_PROCESSING_COMPLETED", created_at: "2026-07-17T10:00:10", session_id: null }),
    ]);

    const runs = await getGraphEnrichmentRuns(instance, DATASET_ID, SESSION);

    expect(runs).toHaveLength(1);
    expect(runs[0].count).toBe(2);
  });

  it("drops an unstamped run once any run is stamped, so an old row does not pose as this session's", async () => {
    const instance = instanceReturning([
      makeRun({ id: "mine", status: "DATASET_PROCESSING_COMPLETED", created_at: "2026-07-17T10:00:00", session_ids: [SESSION] }),
      makeRun({ id: "legacy", status: "DATASET_PROCESSING_ERRORED", created_at: "2026-07-17T10:00:10", session_ids: [] }),
    ]);

    const runs = await getGraphEnrichmentRuns(instance, DATASET_ID, SESSION);

    expect(runs).toHaveLength(1);
    expect(runs[0].count).toBe(1);
    expect(runs[0].status).toBe("completed");
  });

  it("sends no session filter and keeps every run when no session is given", async () => {
    const instance = instanceReturning([
      makeRun({ id: "a", status: "DATASET_PROCESSING_COMPLETED", created_at: "2026-07-17T10:00:00", session_ids: [SESSION] }),
      makeRun({ id: "b", status: "DATASET_PROCESSING_COMPLETED", created_at: "2026-07-17T10:00:10", session_ids: [OTHER] }),
    ]);

    const runs = await getGraphEnrichmentRuns(instance, DATASET_ID);

    expect(requestedUrl(instance)).not.toContain("session_id=");
    expect(runs[0].count).toBe(2);
  });
});

describe("getGraphEnrichmentRuns burst status", () => {
  it("counts an abandoned stage as failed, not as one still running", async () => {
    // The card renders three states only, so the runs feed's ABANDONED has to
    // fold into one of them. Folding it into "running" would leave the card
    // claiming enrichment is in progress for a worker that died.
    const runs = await getGraphEnrichmentRuns(
      instanceReturning([
        makeRun({ id: "a", status: "ABANDONED", created_at: "2026-07-17T10:00:00", error_class: "WorkerLost" }),
      ]),
      DATASET_ID,
    );

    expect(runs).toHaveLength(1);
    expect(runs[0].status).toBe("failed");
    expect(runs[0].error_count).toBe(1);
    // error_class, not error: the endpoint returns only the former, so this
    // assertion used to be fiction. It fed an `error` key no pod sends and
    // passed because the fixture invented the wire shape it was testing.
    expect(runs[0].failure_reason).toBe("WorkerLost");
  });

  it("counts a queued stage as running, since it has not finished", async () => {
    const runs = await getGraphEnrichmentRuns(
      instanceReturning([
        makeRun({ id: "a", status: "DATASET_PROCESSING_INITIATED", created_at: "2026-07-17T10:00:00" }),
      ]),
      DATASET_ID,
    );

    expect(runs).toHaveLength(1);
    expect(runs[0].status).toBe("running");
    expect(runs[0].error_count).toBe(0);
  });

  it("marks a burst completed when some stages errored but others completed", async () => {
    const runs = await getGraphEnrichmentRuns(
      instanceReturning([
        makeRun({ id: "a", status: "DATASET_PROCESSING_COMPLETED", created_at: "2026-07-17T10:02:00" }),
        makeRun({ id: "b", status: "DATASET_PROCESSING_ERRORED", created_at: "2026-07-17T10:01:00" }),
        makeRun({ id: "c", status: "DATASET_PROCESSING_COMPLETED", created_at: "2026-07-17T10:00:00" }),
      ]),
      DATASET_ID,
    );

    expect(runs).toHaveLength(1);
    expect(runs[0].status).toBe("completed");
    expect(runs[0].count).toBe(3);
    expect(runs[0].error_count).toBe(1);
  });

  it("marks a burst completed when the newest stage errored", async () => {
    const runs = await getGraphEnrichmentRuns(
      instanceReturning([
        makeRun({ id: "a", status: "DATASET_PROCESSING_ERRORED", created_at: "2026-07-17T10:01:00" }),
        makeRun({ id: "b", status: "DATASET_PROCESSING_COMPLETED", created_at: "2026-07-17T10:00:00" }),
      ]),
      DATASET_ID,
    );

    expect(runs[0].status).toBe("completed");
    expect(runs[0].error_count).toBe(1);
  });

  it("marks a burst failed only when every stage errored, keeping the newest error as reason", async () => {
    const runs = await getGraphEnrichmentRuns(
      instanceReturning([
        makeRun({ id: "a", status: "DATASET_PROCESSING_ERRORED", created_at: "2026-07-17T10:01:00", error: "PermissionDeniedError: no write access" }),
        makeRun({ id: "b", status: "DATASET_PROCESSING_ERRORED", created_at: "2026-07-17T10:00:00", error: "older error" }),
      ]),
      DATASET_ID,
    );

    expect(runs[0].status).toBe("failed");
    expect(runs[0].error_count).toBe(2);
    expect(runs[0].failure_reason).toBe("PermissionDeniedError: no write access");
  });

  it("tolerates errored rows without an error payload", async () => {
    const runs = await getGraphEnrichmentRuns(
      instanceReturning([
        makeRun({ id: "a", status: "DATASET_PROCESSING_ERRORED", created_at: "2026-07-17T10:00:00" }),
      ]),
      DATASET_ID,
    );

    expect(runs[0].status).toBe("failed");
    expect(runs[0].failure_reason).toBeNull();
  });

  it("marks a burst running while any stage is still in progress", async () => {
    const runs = await getGraphEnrichmentRuns(
      instanceReturning([
        makeRun({ id: "a", status: "DATASET_PROCESSING_STARTED", created_at: "2026-07-17T10:01:00" }),
        makeRun({ id: "b", status: "DATASET_PROCESSING_COMPLETED", created_at: "2026-07-17T10:00:00" }),
      ]),
      DATASET_ID,
    );

    expect(runs[0].status).toBe("running");
  });

  it("bounds the burst duration with started_at (oldest) and created_at (newest)", async () => {
    const runs = await getGraphEnrichmentRuns(
      instanceReturning([
        makeRun({ id: "a", status: "DATASET_PROCESSING_COMPLETED", created_at: "2026-07-17T10:03:30" }),
        makeRun({ id: "b", status: "DATASET_PROCESSING_COMPLETED", created_at: "2026-07-17T10:00:00" }),
      ]),
      DATASET_ID,
    );

    expect(runs[0].created_at).toBe("2026-07-17T10:03:30");
    expect(runs[0].started_at).toBe("2026-07-17T10:00:00");
  });

  it("splits runs further apart than the coalesce window into separate bursts", async () => {
    const runs = await getGraphEnrichmentRuns(
      instanceReturning([
        makeRun({ id: "a", status: "DATASET_PROCESSING_COMPLETED", created_at: "2026-07-17T11:00:00" }),
        makeRun({ id: "b", status: "DATASET_PROCESSING_ERRORED", created_at: "2026-07-17T10:00:00" }),
      ]),
      DATASET_ID,
    );

    expect(runs).toHaveLength(2);
    expect(runs[0].status).toBe("completed");
    expect(runs[1].status).toBe("failed");
  });
});
