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
    fetch: jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(rows),
    }) as unknown as typeof global.fetch,
  };
}

describe("getGraphEnrichmentRuns burst status", () => {
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
