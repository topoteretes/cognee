import type { ReactElement, ReactNode } from "react";
import { renderHook, act, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

// ── Collaborator mocks — declared before module import ────────────────────────

const mockAddFiles = jest.fn();
jest.mock("@/modules/ingestion/addFiles", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockAddFiles(...args),
}));

const mockCognifyDataset = jest.fn();
jest.mock("@/modules/datasets/cognifyDataset", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockCognifyDataset(...args),
}));

const mockPollDatasetStatus = jest.fn();
jest.mock("@/modules/datasets/pollDatasetStatus", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockPollDatasetStatus(...args),
}));

jest.mock("@/modules/datasets/useDatasetStatuses", () => ({
  datasetStatusQueryKey: () => ["dataset-statuses", "tenant-1"],
}));

jest.mock("@tanstack/react-query", () => ({
  ...jest.requireActual("@tanstack/react-query"),
  useQueryClient: () => ({ invalidateQueries: jest.fn() }),
}));

let tenantId: string | undefined = "tenant-1";
jest.mock("@/modules/tenant/TenantProvider", () => ({
  useTenant: () => ({ tenant: tenantId ? { tenant_id: tenantId } : null }),
}));

const mockCaptureException = jest.fn();
jest.mock("@/utils/monitoring", () => ({
  captureException: (...args: unknown[]) => mockCaptureException(...args),
}));

// In-memory stand-in for IndexedDB: jsdom has none, and the real store is
// written to degrade to a no-op there — which would make the resume path
// untestable. This keeps the hook's orchestration under test.
const sessions = new Map<string, Record<string, unknown>>();
let resumable: Record<string, unknown> | null = null;
jest.mock("@/modules/ingestion/uploadSessionStore", () => ({
  // Must mirror the real constant: the hook compares against it, and an
  // undefined here makes every stall-bound test silently vacuous.
  MAX_STALL_MS: 60 * 60 * 1000,
  newSessionId: () => "session-1",
  saveSession: jest.fn(async (session: Record<string, unknown>) => {
    sessions.set(session.id as string, session);
    return session;
  }),
  // Mirrors the real contract: an update never creates. Returns null when the
  // record is gone, which is how a run learns another tab took it over.
  updateSession: jest.fn(async (session: Record<string, unknown>) => {
    if (!sessions.has(session.id as string)) return null;
    sessions.set(session.id as string, session);
    return session;
  }),
  deleteSession: jest.fn(async (id: string) => {
    sessions.delete(id);
  }),
  // A resumable session exists in the store by definition — seed it, or
  // updateSession correctly refuses to write to a record that was never there
  // and every resume test silently stops persisting progress.
  findResumableSession: jest.fn(async () => {
    if (resumable) sessions.set(resumable.id as string, resumable);
    return resumable;
  }),
  listSessions: jest.fn(async () => []),
}));

// ── Module under test ────────────────────────────────────────────────────────

import { useBrainUpload, type BrainUploadParams } from "@/modules/ingestion/useBrainUpload";
import { FILES_PER_BATCH } from "@/modules/ingestion/uploadLimits";
import type { CogneeInstance } from "@/modules/instances/types";

const OLD_TIMEOUT_MS = 5 * 60 * 1000;

function makeFiles(count: number, bytesEach = 1024): File[] {
  return Array.from({ length: count }, (_, i) => {
    const file = new File(["x"], `doc-${i}.txt`, { type: "text/plain" });
    // File.size is read-only in jsdom; define it so byte math is exercised.
    Object.defineProperty(file, "size", { value: bytesEach });
    return file;
  });
}

const instance = { name: "test", fetch: jest.fn() } as unknown as CogneeInstance;

function wrapper({ children }: { children: ReactNode }): ReactElement {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function uploadParams(overrides: Partial<BrainUploadParams> = {}): BrainUploadParams {
  return { datasetId: "ds-1", files: makeFiles(1), ...overrides };
}

beforeEach(() => {
  jest.clearAllMocks();
  sessions.clear();
  resumable = null;
  tenantId = "tenant-1";
  mockAddFiles.mockResolvedValue({ status: "ok" });
  mockCognifyDataset.mockResolvedValue({ pipeline_run_id: "cognify-run-1" });
  mockPollDatasetStatus.mockResolvedValue("DATASET_PROCESSING_COMPLETED");
});

afterEach(() => {
  jest.useRealTimers();
});

describe("batched upload", () => {
  it("splits 200 files into bounded batches instead of one request", async () => {
    const { result } = renderHook(() => useBrainUpload(instance), { wrapper });

    await act(async () => {
      await result.current.upload(uploadParams({ files: makeFiles(200) }));
    });

    expect(mockAddFiles).toHaveBeenCalledTimes(200 / FILES_PER_BATCH);
    for (const call of mockAddFiles.mock.calls) {
      expect((call[1] as File[]).length).toBeLessThanOrEqual(FILES_PER_BATCH);
    }
    // Every selected file went out exactly once.
    const sent = mockAddFiles.mock.calls.flatMap((call) => (call[1] as File[]).map((f) => f.name));
    expect(new Set(sent).size).toBe(200);
  });

  it("completes a 200-file upload that runs far past the old 5-minute window", async () => {
    jest.useFakeTimers();
    jest.setSystemTime(new Date("2026-01-01T00:00:00Z"));

    // Each batch takes 45s of wall clock: 20 batches = 15 simulated minutes,
    // triple the old single-request timeout that made this fail.
    mockAddFiles.mockImplementation(async () => {
      jest.setSystemTime(Date.now() + 45_000);
      return { status: "ok" };
    });

    const onProcessed = jest.fn();
    const onUploadError = jest.fn();
    const { result } = renderHook(() => useBrainUpload(instance), { wrapper });

    await act(async () => {
      await result.current.upload(uploadParams({ files: makeFiles(200), onProcessed, onUploadError }));
    });

    expect(onUploadError).not.toHaveBeenCalled();
    expect(onProcessed).toHaveBeenCalledTimes(1);
    const ctx = onProcessed.mock.calls[0][0];
    expect(ctx.durationMs).toBeGreaterThan(OLD_TIMEOUT_MS);
    expect(ctx.filesUploaded).toBe(200);
  });

  it("reports files-completed and bytes as batches land, not a static spinner", async () => {
    // Batches are held open and released one at a time. With an instantly
    // resolving mock React would batch every state update into a single commit
    // and the hook would appear to jump from idle straight to done — hiding
    // the very thing under test.
    const release: Array<(value: unknown) => void> = [];
    mockAddFiles.mockImplementation(() => new Promise((resolve) => release.push(resolve)));

    const { result } = renderHook(() => useBrainUpload(instance), { wrapper });

    let finished!: Promise<void>;
    await act(async () => {
      finished = result.current.upload(uploadParams({ files: makeFiles(50, 2048) }));
    });

    // BATCH_CONCURRENCY batches are in flight; nothing has landed yet.
    await waitFor(() => expect(release.length).toBeGreaterThan(0));
    expect(result.current.progress.stage).toBe("uploading");
    expect(result.current.progress.filesCompleted).toBe(0);
    expect(result.current.progress.filesTotal).toBe(50);

    const observed: number[] = [];
    while (release.length > 0) {
      await act(async () => {
        release.shift()!({ status: "ok" });
      });
      observed.push(result.current.progress.filesCompleted);
    }
    await act(async () => {
      await finished;
    });

    // Progress climbed in steps rather than jumping 0 → done. The last batch's
    // step commits in the same React batch as the completion reset, so the
    // highest *observable* count is one batch short of the total — the point
    // being that intermediate values exist at all, not their exact tail.
    expect(observed.length).toBeGreaterThan(1);
    expect(new Set(observed).size).toBeGreaterThan(1);
    // Drop the tail: the final sample is taken after completion has already
    // reset the bar to idle, so it reads 0 by design.
    const climbing = observed.slice(0, -1);
    expect(climbing).toEqual([...climbing].sort((a, b) => a - b));
    expect(Math.max(...observed)).toBeGreaterThanOrEqual(50 - FILES_PER_BATCH);
    expect(mockAddFiles).toHaveBeenCalledTimes(5);
  });

  it("does not report a total failure when earlier batches already landed", async () => {
    let call = 0;
    mockAddFiles.mockImplementation(async () => {
      call += 1;
      if (call === 3) throw new Error("network blip");
      return { status: "ok" };
    });

    const onUploadError = jest.fn();
    const { result } = renderHook(() => useBrainUpload(instance), { wrapper });

    await act(async () => {
      await result.current.upload(uploadParams({ files: makeFiles(50), onUploadError }));
    });

    expect(onUploadError).toHaveBeenCalledTimes(1);
    const ctx = onUploadError.mock.calls[0][1];
    // The files from the batches that DID land are durable — the callback must
    // say so, or the UI would tell the user everything was lost.
    expect(ctx.filesUploaded).toBeGreaterThan(0);
    expect(ctx.filesUploaded).toBeLessThan(50);
  });
});

describe("one run at a time", () => {
  it("refuses a second upload while one is in flight instead of corrupting progress", async () => {
    const release: Array<(value: unknown) => void> = [];
    mockAddFiles.mockImplementation(() => new Promise((resolve) => release.push(resolve)));

    const { result } = renderHook(() => useBrainUpload(instance), { wrapper });

    let first!: Promise<void>;
    await act(async () => {
      first = result.current.upload(uploadParams({ files: makeFiles(30) }));
    });
    await waitFor(() => expect(release.length).toBeGreaterThan(0));

    const onUploadError = jest.fn();
    await act(async () => {
      await result.current.upload(uploadParams({ files: makeFiles(5), onUploadError }));
    });

    expect(onUploadError).toHaveBeenCalledTimes(1);
    expect(String(onUploadError.mock.calls[0][0])).toMatch(/already in progress/i);
    // The rejected call must not have reset the running upload's totals.
    expect(result.current.progress.filesTotal).toBe(30);

    while (release.length > 0) {
      await act(async () => {
        release.shift()!({ status: "ok" });
      });
    }
    await act(async () => {
      await first;
    });
  });

  it("does not resurrect a session another tab has taken over", async () => {
    const release: Array<(value: unknown) => void> = [];
    mockAddFiles.mockImplementation(() => new Promise((resolve) => release.push(resolve)));

    const { result } = renderHook(() => useBrainUpload(instance), { wrapper });
    let finished!: Promise<void>;
    await act(async () => {
      finished = result.current.upload(uploadParams({ files: makeFiles(30) }));
    });
    await waitFor(() => expect(release.length).toBeGreaterThan(0));
    expect(sessions.has("session-1")).toBe(true);

    // Another tab starts its own upload for this dataset and drops ours.
    sessions.delete("session-1");

    // Land one batch so the run performs a progress write, and assert on that
    // write specifically — not on the end of the run, which deletes the
    // session anyway and would hide a resurrection behind the cleanup.
    const { updateSession } = jest.requireMock("@/modules/ingestion/uploadSessionStore");
    const writesBefore = updateSession.mock.calls.length;
    await act(async () => {
      release.shift()!({ status: "ok" });
    });
    await waitFor(() => expect(updateSession.mock.calls.length).toBeGreaterThan(writesBefore));

    // IndexedDB's put() inserts when the key is absent, so saving progress
    // through saveSession would bring the deleted record back and leave two
    // sessions competing for one dataset. Updates must be updates.
    expect(sessions.has("session-1")).toBe(false);

    // Let the run unwind so it doesn't leak into the next test.
    while (release.length > 0) {
      await act(async () => {
        release.shift()!({ status: "ok" });
      });
    }
    await act(async () => {
      await finished;
    });

    // Losing the session is silent to the user by design — the files still
    // land — but it must not be silent to us: from that point the run cannot
    // be resumed. Reported once, not once per batch.
    const takeovers = mockCaptureException.mock.calls.filter(([error]) =>
      String((error as Error)?.message).includes("taken over by another tab"),
    );
    expect(takeovers).toHaveLength(1);
    expect(takeovers[0][1]).toMatchObject({ datasetId: "ds-1", sessionId: "session-1" });
  });

  it("drops a stale session for the same dataset before starting a new one", async () => {
    const { deleteSession } = jest.requireMock("@/modules/ingestion/uploadSessionStore");

    // Null at mount so auto-resume stays out of this; the stale record only
    // becomes visible to the lookup upload() itself performs.
    resumable = null;
    const { result } = renderHook(() => useBrainUpload(instance), { wrapper });
    await waitFor(() => expect(result.current.isUploading).toBe(false));

    resumable = { id: "stale-session", datasetId: "ds-1", pending: [], filesTotal: 3 };
    await act(async () => {
      await result.current.upload(uploadParams({ files: makeFiles(5) }));
    });

    // Specifically the stale one — the run also deletes its OWN session on
    // completion, so a bare "was called" assertion would pass without the fix.
    expect(deleteSession).toHaveBeenCalledWith("stale-session");
  });
});

describe("state released on failure", () => {
  it("lets a later resume run after a failed upload instead of blocking on it", async () => {
    mockAddFiles.mockRejectedValue(new Error("network blip"));

    const onUploadError = jest.fn();
    const { result, rerender } = renderHook(
      ({ inst }: { inst: CogneeInstance }) => useBrainUpload(inst),
      { wrapper, initialProps: { inst: instance } },
    );
    await act(async () => {
      await result.current.upload(uploadParams({ files: makeFiles(30), onUploadError }));
    });
    expect(onUploadError).toHaveBeenCalled();

    // The session is kept on purpose so the remainder can be picked up. But the
    // hook also held a reference to it, and the resume effect bails when
    // activeSession.current is set — so the very session that was preserved
    // could never be resumed while this component stayed mounted.
    mockAddFiles.mockReset();
    mockAddFiles.mockResolvedValue({ status: "ok" });
    resumable = {
      id: "session-after-failure",
      tenantId: "tenant-1",
      datasetId: "ds-1",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      lastProgressAt: Date.now(),
      filesTotal: 30,
      bytesTotal: 30 * 1024,
      filesCompleted: 10,
      bytesCompleted: 10 * 1024,
      pending: makeFiles(20),
      runIds: [],
      stage: "uploading",
    };

    // A NEW instance object, so the resume effect's dependencies actually
    // change and it runs again — rerender() alone leaves the deps identical
    // and the effect never fires, which makes this assertion meaningless.
    await act(async () => {
      rerender({ inst: { ...instance } as CogneeInstance });
    });

    // The preserved remainder goes out. Without releasing activeSession.current
    // the effect returns early here and the upload is silently abandoned.
    await waitFor(() => expect(mockAddFiles).toHaveBeenCalled());
    const sent = mockAddFiles.mock.calls.flatMap((c) => (c[1] as File[]).map((f) => f.name));
    expect(sent).toHaveLength(20);
  });

  it("reports why a resumed upload stopped instead of failing silently", async () => {
    resumable = {
      id: "session-1",
      tenantId: "tenant-1",
      datasetId: "ds-1",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      lastProgressAt: Date.now(),
      filesTotal: 20,
      bytesTotal: 20 * 1024,
      filesCompleted: 0,
      bytesCompleted: 0,
      pending: makeFiles(20),
      runIds: [],
      stage: "uploading",
    };
    mockAddFiles.mockRejectedValue(new Error("401 token expired"));

    renderHook(() => useBrainUpload(instance), { wrapper });

    // The resume path has no caller and so no onUploadError to raise. A bare
    // catch made an expired token look exactly like a network blip: the upload
    // just stopped, with nothing recorded anywhere.
    await waitFor(() => {
      const reported = mockCaptureException.mock.calls.filter(
        ([, ctx]) => (ctx as { stage?: string })?.stage === "resume",
      );
      expect(reported).toHaveLength(1);
      expect(String((reported[0][0] as Error).message)).toContain("401");
    });
  });

  it("reports a build that fails on a resumed upload instead of swallowing it", async () => {
    const { deleteSession } = jest.requireMock("@/modules/ingestion/uploadSessionStore");
    resumable = {
      id: "session-1",
      tenantId: "tenant-1",
      datasetId: "ds-1",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      lastProgressAt: Date.now(),
      filesTotal: 20,
      bytesTotal: 20 * 1024,
      filesCompleted: 10,
      bytesCompleted: 10 * 1024,
      pending: makeFiles(10),
      runIds: [],
      stage: "uploading",
    };
    mockCognifyDataset.mockRejectedValue(new Error("cognify refused"));

    const { result } = renderHook(() => useBrainUpload(instance), { wrapper });
    await waitFor(() => expect(deleteSession).toHaveBeenCalledWith("session-1"));
    await waitFor(() => expect(result.current.isUploading).toBe(false));

    // The resume path has no caller and so no onProcessingError to raise.
    // Empty hooks meant the cognify failure vanished along with the session —
    // files durable, graph never built, and no trace of either.
    const reported = mockCaptureException.mock.calls.filter(
      ([, ctx]) => (ctx as { stage?: string })?.stage === "resume-processing",
    );
    expect(reported).toHaveLength(1);
    expect(String((reported[0][0] as Error).message)).toContain("cognify refused");
    // The count is what actually landed by then, not what the record said at
    // pickup: the 10 pending files were replayed before the build was started.
    expect(reported[0][1]).toMatchObject({ datasetId: "ds-1", sessionId: "session-1", filesUploaded: 20, filesTotal: 20 });
    expect(mockPollDatasetStatus).not.toHaveBeenCalled();
  });

  it("does not report a deliberate cancel of a resumed upload as a failure", async () => {
    resumable = {
      id: "session-1",
      tenantId: "tenant-1",
      datasetId: "ds-1",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      lastProgressAt: Date.now(),
      filesTotal: 20,
      bytesTotal: 20 * 1024,
      filesCompleted: 0,
      bytesCompleted: 0,
      pending: makeFiles(20),
      runIds: [],
      stage: "uploading",
    };
    const abortErr = new Error("Upload cancelled");
    abortErr.name = "AbortError";
    mockAddFiles.mockRejectedValue(abortErr);

    const { result } = renderHook(() => useBrainUpload(instance), { wrapper });
    await waitFor(() => expect(result.current.isUploading).toBe(false));

    expect(
      mockCaptureException.mock.calls.filter(([, ctx]) => (ctx as { stage?: string })?.stage === "resume"),
    ).toHaveLength(0);
  });
});

describe("tenant not ready", () => {
  it("refuses an upload rather than writing a session no lookup can find", async () => {
    const { saveSession } = jest.requireMock("@/modules/ingestion/uploadSessionStore");
    tenantId = undefined;

    const onUploadError = jest.fn();
    const { result } = renderHook(() => useBrainUpload(instance), { wrapper });
    await act(async () => {
      await result.current.upload(uploadParams({ files: makeFiles(5), onUploadError }));
    });

    // A session stamped with a placeholder tenant is unreachable the moment the
    // page reloads — listSessions matches on tenant id — so its pending files
    // would be stranded until MAX_AGE_MS collected them.
    expect(saveSession).not.toHaveBeenCalled();
    expect(mockAddFiles).not.toHaveBeenCalled();
    // And it says so, rather than a click doing nothing.
    expect(onUploadError).toHaveBeenCalledTimes(1);
    expect(String(onUploadError.mock.calls[0][0])).toMatch(/workspace/i);
  });
});

describe("cancellation", () => {
  it("never reports a cancelled run as a completed upload", async () => {
    const release: Array<(value: unknown) => void> = [];
    mockAddFiles.mockImplementation(() => new Promise((resolve) => release.push(resolve)));

    const onUploaded = jest.fn();
    const onProcessed = jest.fn();
    const onUploadError = jest.fn();
    const { result } = renderHook(() => useBrainUpload(instance), { wrapper });

    let finished!: Promise<void>;
    await act(async () => {
      finished = result.current.upload(
        uploadParams({ files: makeFiles(50), onUploaded, onProcessed, onUploadError }),
      );
    });
    await waitFor(() => expect(release.length).toBeGreaterThan(0));

    await act(async () => {
      result.current.cancel();
      // Let the in-flight batches settle so the workers observe the abort.
      while (release.length > 0) release.shift()!({ status: "ok" });
      await finished;
    });

    // The bug: workers returned on `signal.aborted` without recording it, so
    // sendBatches resolved, the caller counted every file as accepted, and
    // onUploaded fired for files that never left the browser.
    expect(onUploaded).not.toHaveBeenCalled();
    expect(onProcessed).not.toHaveBeenCalled();
    expect(mockPollDatasetStatus).not.toHaveBeenCalled();
    // A cancel is not a failure either — no error banner for a deliberate stop.
    expect(onUploadError).not.toHaveBeenCalled();
  });

  it("aborts the in-flight run on unmount so the next mount can take it over", async () => {
    const signals: AbortSignal[] = [];
    const release: Array<(value: unknown) => void> = [];
    mockAddFiles.mockImplementation((_ds, _files, _inst, opts) => {
      if (opts?.signal) signals.push(opts.signal as AbortSignal);
      return new Promise((resolve) => release.push(resolve));
    });

    const { result, unmount } = renderHook(() => useBrainUpload(instance), { wrapper });
    await act(async () => {
      void result.current.upload(uploadParams({ files: makeFiles(50) }));
    });
    await waitFor(() => expect(signals.length).toBeGreaterThan(0));
    expect(signals[0].aborted).toBe(false);

    unmount();

    // Without this the unmounting instance keeps sending while the next mount
    // resumes the same session — the same files ingested twice.
    expect(signals[0].aborted).toBe(true);
  });
});

describe("resume after refresh", () => {
  it("replays only the pending remainder and never re-sends what already landed", async () => {
    const pending = makeFiles(20);
    resumable = {
      id: "session-1",
      tenantId: "tenant-1",
      datasetId: "ds-1",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      filesTotal: 100,
      bytesTotal: 100 * 1024,
      filesCompleted: 80,
      bytesCompleted: 80 * 1024,
      pending,
      runIds: [],
      stage: "uploading",
    };

    renderHook(() => useBrainUpload(instance), { wrapper });

    await waitFor(() => expect(mockAddFiles).toHaveBeenCalled());
    await waitFor(() => expect(mockPollDatasetStatus).toHaveBeenCalled());

    const sent = mockAddFiles.mock.calls.flatMap((c) => (c[1] as File[]).map((f) => f.name));
    expect(sent).toHaveLength(20);
    expect(new Set(sent)).toEqual(new Set(pending.map((f) => f.name)));
  });

  it("drops a session that has made no progress for longer than the stall window", async () => {
    const { deleteSession } = jest.requireMock("@/modules/ingestion/uploadSessionStore");
    resumable = {
      id: "doomed-session",
      tenantId: "tenant-1",
      datasetId: "ds-1",
      createdAt: Date.now() - 3 * 60 * 60 * 1000,
      updatedAt: Date.now(),
      // Last actually moved three hours ago — genuinely wedged, not merely
      // reloaded a few times.
      lastProgressAt: Date.now() - 3 * 60 * 60 * 1000,
      filesTotal: 10,
      bytesTotal: 10 * 1024,
      filesCompleted: 0,
      bytesCompleted: 0,
      pending: makeFiles(10),
      runIds: [],
      stage: "uploading",
    };

    renderHook(() => useBrainUpload(instance), { wrapper });

    await waitFor(() => expect(deleteSession).toHaveBeenCalledWith("doomed-session"));
    expect(mockAddFiles).not.toHaveBeenCalled();
    expect(mockPollDatasetStatus).not.toHaveBeenCalled();
  });

  it("keeps resuming a slow upload no matter how many times it is reloaded", async () => {
    const { deleteSession } = jest.requireMock("@/modules/ingestion/uploadSessionStore");
    // The case an attempt counter got wrong: a big upload on a slow link that
    // the user has reloaded past many times. It is progressing — the last
    // batch landed a minute ago — so it must not be dropped.
    resumable = {
      id: "slow-session",
      tenantId: "tenant-1",
      datasetId: "ds-1",
      createdAt: Date.now() - 6 * 60 * 60 * 1000,
      updatedAt: Date.now(),
      lastProgressAt: Date.now() - 60 * 1000,
      filesTotal: 200,
      bytesTotal: 200 * 1024,
      filesCompleted: 120,
      bytesCompleted: 120 * 1024,
      pending: makeFiles(80),
      runIds: [],
      stage: "uploading",
    };

    // Hold polling open so the run doesn't reach the completion path, which
    // legitimately deletes the session — that would mask the drop under test.
    mockPollDatasetStatus.mockImplementation(() => new Promise(() => {}));

    renderHook(() => useBrainUpload(instance), { wrapper });

    // It was picked up and sent rather than dropped for having been reloaded.
    await waitFor(() => expect(mockAddFiles).toHaveBeenCalled());
    expect(deleteSession).not.toHaveBeenCalledWith("slow-session");
  });

  it("refreshes the stall clock on every landed batch", async () => {
    const { updateSession } = jest.requireMock("@/modules/ingestion/uploadSessionStore");
    const before = Date.now();

    const { result } = renderHook(() => useBrainUpload(instance), { wrapper });
    await act(async () => {
      await result.current.upload(uploadParams({ files: makeFiles(30) }));
    });

    const stamps = updateSession.mock.calls
      .map((call: [Record<string, unknown>]) => call[0].lastProgressAt as number | undefined)
      .filter((v: number | undefined): v is number => typeof v === "number");
    expect(stamps.length).toBeGreaterThan(0);
    expect(Math.max(...stamps)).toBeGreaterThanOrEqual(before);
  });

  it("records the pipeline run id of every landed batch", async () => {
    const { updateSession } = jest.requireMock("@/modules/ingestion/uploadSessionStore");
    let n = 0;
    mockAddFiles.mockImplementation(async () => {
      n += 1;
      return { status: "ok", pipeline_run_id: `run-${n}` };
    });

    const { result } = renderHook(() => useBrainUpload(instance), { wrapper });
    await act(async () => {
      await result.current.upload(uploadParams({ files: makeFiles(30) }));
    });

    // runIds is the only handle a backend log search has on these batches; it
    // was documented as the support hook while never being written to.
    const runIds = updateSession.mock.calls
      .map((call: [Record<string, unknown>]) => call[0].runIds as string[] | undefined)
      .filter((v: string[] | undefined): v is string[] => Array.isArray(v) && v.length > 0)
      .pop();
    expect(runIds).toEqual(["run-1", "run-2", "run-3"]);
  });

  it("does not report a quota-degraded session as a completed upload", async () => {
    // saveSession drops the blobs when the browser refuses to store them, so
    // the record has an empty `pending` while files are still unsent. Reading
    // only `pending.length` made that indistinguishable from "all done": the
    // resume skipped the upload and polled as though 100 files had landed.
    resumable = {
      id: "degraded-session",
      tenantId: "tenant-1",
      datasetId: "ds-1",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      lastProgressAt: Date.now(),
      filesTotal: 100,
      bytesTotal: 100 * 1024,
      filesCompleted: 30,
      bytesCompleted: 30 * 1024,
      pending: [],
      degraded: true,
      runIds: [],
      stage: "uploading",
    };

    // Hold polling open: awaitProcessing's finally resets progress to idle the
    // moment it resolves, which would erase what this asserts.
    mockPollDatasetStatus.mockImplementation(() => new Promise(() => {}));

    const { result } = renderHook(() => useBrainUpload(instance), { wrapper });

    await waitFor(() => expect(result.current.progress.unrecoverableFiles).toBe(70));
    // The 30 that landed are real and still worth polling for.
    expect(result.current.progress.filesCompleted).toBe(30);
    expect(mockAddFiles).not.toHaveBeenCalled();
  });

  it("restores file types on resume instead of reporting none", async () => {
    resumable = {
      id: "session-1",
      tenantId: "tenant-1",
      datasetId: "ds-1",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      lastProgressAt: Date.now(),
      filesTotal: 20,
      bytesTotal: 20 * 1024,
      filesCompleted: 10,
      bytesCompleted: 10 * 1024,
      pending: makeFiles(10),
      fileTypes: ["application/pdf", "text/plain"],
      runIds: [],
      stage: "uploading",
    };

    renderHook(() => useBrainUpload(instance), { wrapper });
    await waitFor(() => expect(mockPollDatasetStatus).toHaveBeenCalled());

    // A resumed run holds no File objects for what already landed, so the
    // types have to come off the session or analytics sees an empty list.
    const { saveSession } = jest.requireMock("@/modules/ingestion/uploadSessionStore");
    expect(saveSession).not.toHaveBeenCalledWith(expect.objectContaining({ fileTypes: [] }));
    expect(resumable.fileTypes).toEqual(["application/pdf", "text/plain"]);
  });

  it("goes straight to polling when nothing is left to send", async () => {
    resumable = {
      id: "session-1",
      tenantId: "tenant-1",
      datasetId: "ds-1",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      filesTotal: 100,
      bytesTotal: 100 * 1024,
      filesCompleted: 100,
      bytesCompleted: 100 * 1024,
      pending: [],
      runIds: [],
      stage: "processing",
    };

    renderHook(() => useBrainUpload(instance), { wrapper });

    await waitFor(() => expect(mockPollDatasetStatus).toHaveBeenCalledWith("ds-1", instance, expect.anything()));
    expect(mockAddFiles).not.toHaveBeenCalled();
  });
});
