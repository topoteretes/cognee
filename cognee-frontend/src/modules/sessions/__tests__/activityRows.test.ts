import {
  activityRowStatus,
  countsTowardTotals,
  dedupeByRunId,
  normalizeActivityRows,
  runStatus,
  sessionIdsWithOperationRows,
  viaChannel,
  viaLabel,
  type PipelineRun,
} from "../getSessions";

/** The 9-key row shape a pod predating SDK-399 actually returns. */
const LEGACY_ROW = {
  id: "id-1",
  pipeline_name: "cognify_pipeline",
  status: "DATASET_PROCESSING_COMPLETED",
  dataset_id: "ds-1",
  dataset_name: "Notes",
  owner_id: "user-1",
  owner_email: "a@example.com",
  created_at: "2026-09-01T10:00:00",
  pipeline_run_id: "run-1",
};

/** Overrides are the raw wire shape, not PipelineRun: the point of these
 *  tests is what arrives before normalization, including a null `status`,
 *  which only operation rows have. */
function row(overrides: Record<string, unknown> = {}): PipelineRun {
  return normalizeActivityRows([{ ...LEGACY_ROW, ...overrides }])[0];
}

describe("normalizeActivityRows", () => {
  it("throws rather than reading a non-array payload as an empty workspace", () => {
    // A 200 carrying an error envelope was previously indistinguishable from
    // a tenant that had never done anything.
    expect(() => normalizeActivityRows({ error: "boom" })).toThrow(/non-array/);
    expect(() => normalizeActivityRows(null)).toThrow(/non-array/);
    expect(() => normalizeActivityRows("[]")).toThrow(/non-array/);
  });

  it("accepts an empty array", () => {
    expect(normalizeActivityRows([])).toEqual([]);
  });

  it("turns every absent SDK-399 column into an explicit null", () => {
    const [r] = normalizeActivityRows([LEGACY_ROW]);
    // `=== null` comparisons downstream (CostPanel, the Activity footer) are
    // only correct because the boundary does this.
    expect(r.tokens_in).toBeNull();
    expect(r.tokens_out).toBeNull();
    expect(r.parent_operation_id).toBeNull();
    expect(r.session_ids).toEqual([]);
    expect(r.origin).toBeNull();
    expect(r.outcome).toBeNull();
    expect(r.background).toBeNull();
    expect(r.user_id).toBeNull();
    expect(r.operation_name).toBeNull();
  });

  it("derives an absent kind from pipeline_name, it does not default it", () => {
    // The premise this replaces was false: cognee has written operation rows
    // since v1.5.0.dev3 while the router only started returning `kind` in
    // 1.5.3.dev1, so a pod that omits the field is still sending operations.
    // Defaulting them to "pipeline" made a forget render as "remember".
    expect(normalizeActivityRows([LEGACY_ROW])[0].kind).toBe("pipeline");
    const noPipelineName = { ...LEGACY_ROW } as Record<string, unknown>;
    delete noPipelineName.pipeline_name;
    expect(normalizeActivityRows([noPipelineName])[0].kind).toBe("operation");
    expect(normalizeActivityRows([{ ...LEGACY_ROW, pipeline_name: null }])[0].kind).toBe("operation");
  });

  it("rejects a kind outside the union rather than letting it fall into the pipeline arm", () => {
    // The type promises consumers can branch on `kind` and keep working if a
    // third kind appears. This is the one place that can enforce it.
    const [r] = normalizeActivityRows([{ ...LEGACY_ROW, pipeline_name: null, kind: "trace" }]);
    expect(r.kind).toBe("operation");
  });

  it("preserves the values a modern row does carry", () => {
    const [r] = normalizeActivityRows([
      { ...LEGACY_ROW, kind: "operation", tokens_in: 0, tokens_out: 12, background: false },
    ]);
    expect(r.kind).toBe("operation");
    // Zero is a measurement, not an absence — `?? null` must not eat it.
    expect(r.tokens_in).toBe(0);
    expect(r.tokens_out).toBe(12);
    expect(r.background).toBe(false);
  });

  it("skips entries that are not objects instead of throwing on the whole page", () => {
    expect(normalizeActivityRows([LEGACY_ROW, null, "x", 7]).length).toBe(1);
  });
});

describe("normalizeActivityRows session_ids", () => {
  it("keeps the list a pod running cognee PR 6191 sends, in its order", () => {
    const [r] = normalizeActivityRows([{ ...LEGACY_ROW, session_ids: ["claude_a", "cursor_b"] }]);
    expect(r.session_ids).toEqual(["claude_a", "cursor_b"]);
  });

  it("folds the single session_id of an older pod into a one-element list", () => {
    // Pods from before PR 6191 (including ones running PR 6160) send
    // `session_id` and no list; nothing downstream should branch on which.
    const [r] = normalizeActivityRows([{ ...LEGACY_ROW, session_id: "claude_a" }]);
    expect(r.session_ids).toEqual(["claude_a"]);
  });

  it("prefers the list when a row carries both, even an empty one", () => {
    // A row never half-carries its sessions: the list is the pod's statement.
    const [r] = normalizeActivityRows([{ ...LEGACY_ROW, session_ids: [], session_id: "claude_a" }]);
    expect(r.session_ids).toEqual([]);
  });

  it("reads an explicit null and a non-list as no sessions", () => {
    expect(normalizeActivityRows([{ ...LEGACY_ROW, session_id: null }])[0].session_ids).toEqual([]);
    expect(normalizeActivityRows([{ ...LEGACY_ROW, session_ids: "claude_a" }])[0].session_ids).toEqual([]);
  });

  it("drops entries that are not session ids rather than letting them reach the channel lookup", () => {
    const [r] = normalizeActivityRows([{ ...LEGACY_ROW, session_ids: ["claude_a", 7, null, ""] }]);
    expect(r.session_ids).toEqual(["claude_a"]);
  });
});

describe("normalizeActivityRows status typing", () => {
  it("does not let a non-string status reach the status readers", () => {
    // The documented producer sends `string | null`, so this is hardening
    // rather than a fix. It is cheap because the failure mode is not one bad
    // row: `runStatus` calls `.includes`, and a throw inside buildActivityRows
    // runs in a plain useMemo, so it blanks the whole Activity page, dashboard
    // panel and analytics page.
    for (const bad of [42, true, {}, []]) {
      const [row] = normalizeActivityRows([{ ...LEGACY_ROW, status: bad }]);
      expect(typeof row.status).toBe("string");
      expect(() => activityRowStatus(row)).not.toThrow();
    }
  });
});

describe("runStatus", () => {
  it("reads the four stored PipelineRunStatus members", () => {
    expect(runStatus("DATASET_PROCESSING_COMPLETED")).toBe("completed");
    expect(runStatus("DATASET_PROCESSING_ERRORED")).toBe("failed");
    expect(runStatus("DATASET_PROCESSING_STARTED")).toBe("running");
    // A run queued by a reset or a recovery: no started_at, nothing in flight.
    expect(runStatus("DATASET_PROCESSING_INITIATED")).toBe("queued");
  });

  it("reads ABANDONED, which the endpoint computes rather than stores", () => {
    // cognee SDK-591: a row stuck at STARTED past the threshold is reported
    // as ABANDONED. This used to fall through to "running", which is why the
    // feed showed crashed runs as in flight with an ever-growing duration.
    expect(runStatus("ABANDONED")).toBe("abandoned");
  });

  it("still falls back to running for a status it has not been taught", () => {
    // The one honest default: a live pod can report a member added after this
    // build shipped, and the rows that carry it are the ones in flight.
    expect(runStatus("DATASET_PROCESSING_SOMETHING_NEW")).toBe("running");
    expect(runStatus(undefined)).toBe("running");
  });
});

describe("dedupeByRunId", () => {
  it("keeps one row per pipeline_run_id", () => {
    // A run emits a started row and a terminal row under one run id.
    const rows = normalizeActivityRows([
      { ...LEGACY_ROW, id: "a", status: "DATASET_PROCESSING_STARTED" },
      { ...LEGACY_ROW, id: "b", status: "DATASET_PROCESSING_COMPLETED" },
    ]);
    expect(dedupeByRunId(rows).length).toBe(1);
  });

  it("does not collapse distinct runs", () => {
    const rows = normalizeActivityRows([
      { ...LEGACY_ROW, pipeline_run_id: "run-1" },
      { ...LEGACY_ROW, pipeline_run_id: "run-2" },
    ]);
    expect(dedupeByRunId(rows).length).toBe(2);
  });

  it("prefers a real terminal row over the read-time abandoned verdict", () => {
    // A straggler worker that finishes past the abandon threshold writes a
    // COMPLETED row sharing the id, and that row is the truth. The endpoint
    // keeps reporting the older STARTED row as ABANDONED forever.
    const rows = normalizeActivityRows([
      { ...LEGACY_ROW, id: "a", status: "ABANDONED", created_at: "2026-09-01T10:00:00" },
      { ...LEGACY_ROW, id: "b", status: "DATASET_PROCESSING_COMPLETED", created_at: "2026-09-01T12:00:00" },
    ]);
    expect(dedupeByRunId(rows).map((r) => r.id)).toEqual(["b"]);
  });

  it("prefers the completion even when the abandoned verdict is newer", () => {
    // The one input that distinguishes ranking abandoned below a terminal row
    // from ranking it level with one. Reachable only on a pod old enough to
    // derive pipeline_run_id from pipeline_id + dataset_id, where run N's stuck
    // STARTED row reads ABANDONED and is newer than run N-1's COMPLETED row on
    // the same id. Rank it level and the newer-wins tie-break would serve the
    // abandoned verdict instead.
    const rows = normalizeActivityRows([
      { ...LEGACY_ROW, id: "completed", status: "DATASET_PROCESSING_COMPLETED", created_at: "2026-09-01T10:00:00" },
      { ...LEGACY_ROW, id: "abandoned", status: "ABANDONED", created_at: "2026-09-01T12:00:00" },
    ]);
    expect(dedupeByRunId(rows).map((r) => r.id)).toEqual(["completed"]);
  });

  it("does not let a queued row outrank the started row it precedes", () => {
    // Reachable on a pod old enough to derive pipeline_run_id deterministically
    // from pipeline_id + dataset_id, where the INITIATED row does share the id.
    // It becomes the ordinary case once SDK-595 lands, since the reset writer
    // will then share the id on every pod. See CLO-756.
    // The finality test this replaced was `!== "running"`, which read a queued
    // row as terminal and let it win.
    const rows = normalizeActivityRows([
      { ...LEGACY_ROW, id: "a", status: "DATASET_PROCESSING_INITIATED", created_at: "2026-09-01T10:00:00" },
      { ...LEGACY_ROW, id: "b", status: "DATASET_PROCESSING_STARTED", created_at: "2026-09-01T10:00:05" },
    ]);
    expect(dedupeByRunId(rows).map((r) => r.id)).toEqual(["b"]);
  });

  it("keeps rows with no run id apart, since they share no identity", () => {
    const rows = normalizeActivityRows([
      { ...LEGACY_ROW, id: "a", pipeline_run_id: null },
      { ...LEGACY_ROW, id: "b", pipeline_run_id: null },
    ]);
    expect(dedupeByRunId(rows).length).toBe(2);
  });

  it("keeps rows apart even with neither a run id nor an id", () => {
    // The uniqueness fallback used `??`, which an empty-string id from the
    // normalizer never reaches, so every such row collapsed onto one key.
    // Unreachable from a real response (id is a non-nullable primary key),
    // but a safeguard that cannot fire is worse than none.
    const rows = normalizeActivityRows([
      { ...LEGACY_ROW, id: undefined, pipeline_run_id: null },
      { ...LEGACY_ROW, id: undefined, pipeline_run_id: null },
    ]);
    expect(dedupeByRunId(rows).length).toBe(2);
  });
});

describe("sessionIdsWithOperationRows", () => {
  it("reports the session an operation row references", () => {
    const rows = [row({ kind: "operation", session_ids: ["search-ui-1"] })];
    expect(sessionIdsWithOperationRows(rows).has("search-ui-1")).toBe(true);
  });

  it("reports every session of an operation over several, so each is covered", () => {
    // The gap this closes: an improve over two sessions used to record no
    // session at all, so both sessions kept their rows and the same recall
    // rendered twice.
    const rows = [row({ kind: "operation", session_ids: ["search-ui-1", "search-ui-2"] })];
    const ids = sessionIdsWithOperationRows(rows);
    expect(ids.has("search-ui-1")).toBe(true);
    expect(ids.has("search-ui-2")).toBe(true);
  });

  it("still reads the single session an older pod sends", () => {
    const rows = [row({ kind: "operation", session_id: "search-ui-1" })];
    expect(sessionIdsWithOperationRows(rows).has("search-ui-1")).toBe(true);
  });

  it("ignores a pipeline row's session, which does not cover a session row", () => {
    const rows = [row({ kind: "pipeline", session_ids: ["search-ui-1"] })];
    expect(sessionIdsWithOperationRows(rows).size).toBe(0);
  });

  it("is empty for a legacy feed, so its session rows survive", () => {
    // The regression this replaces: suppressing session rows on the pod's
    // mere capability to emit operations erased pre-SDK-399 history.
    expect(sessionIdsWithOperationRows(normalizeActivityRows([LEGACY_ROW])).size).toBe(0);
  });
});

describe("activityRowStatus", () => {
  it("reads outcome for an operation row, which has status: null", () => {
    expect(activityRowStatus(row({ status: null, outcome: "failed" }))).toBe("failed");
    expect(activityRowStatus(row({ status: null, outcome: "succeeded" }))).toBe("completed");
  });

  it("calls a succeeded background operation accepted, not running", () => {
    // background: true means the work was handed off, not finished. It is not
    // "running" either: nothing ever revisits this row, so "running" was a
    // state the row could never leave. The launched pipelines carry the real
    // progress.
    expect(activityRowStatus(row({ status: null, outcome: "succeeded", background: true }))).toBe("accepted");
  });

  it("falls back to the pipeline status when there is no outcome", () => {
    expect(activityRowStatus(row({ status: "DATASET_PROCESSING_COMPLETED" }))).toBe("completed");
  });

  it("says unknown for an operation row with neither status nor outcome", () => {
    // What a released pod actually returns: it omits `outcome` in the same
    // versions that omit `kind`, and calling a finished forget "running"
    // forever is a worse lie than admitting the row does not say.
    const [r] = normalizeActivityRows([{ ...LEGACY_ROW, pipeline_name: null, status: null }]);
    expect(r.kind).toBe("operation");
    expect(activityRowStatus(r)).toBe("unknown");
  });

  it("still reports a genuinely in-flight pipeline run as running", () => {
    expect(activityRowStatus(row({ status: "DATASET_PROCESSING_STARTED" }))).toBe("running");
  });
});

describe("countsTowardTotals", () => {
  it("excludes a nested row, whose tokens are already in its parent", () => {
    expect(countsTowardTotals(row({ parent_operation_id: "op-1" }))).toBe(false);
  });

  it("includes a top-level row", () => {
    expect(countsTowardTotals(row({ parent_operation_id: null }))).toBe(true);
  });

  it("includes a legacy row, which omits the column entirely", () => {
    // `== null`, not `=== null`: treating undefined as "has a parent" would
    // zero every total on an old pod.
    expect(countsTowardTotals({ parent_operation_id: undefined } as unknown as PipelineRun)).toBe(true);
  });
});

describe("viaLabel", () => {
  it("names the dashboard Cloud, not UI", () => {
    expect(viaLabel(row({ session_id: "search-ui-abc" }))).toBe("Cloud");
  });

  it("names a known client from its session prefix", () => {
    expect(viaLabel(row({ session_id: "claude_123" }))).toBe("Claude Code");
    expect(viaLabel(row({ session_id: "claude_desktop_123" }))).toBe("Claude Desktop");
  });

  it("prefers the session over origin, which is one value for every pod caller", () => {
    expect(viaLabel(row({ session_id: "cursor_1", origin: "api" }))).toBe("Cursor");
  });

  it("discards origin: sdk, which is the default-when-unset every pod row carries", () => {
    expect(viaLabel(row({ session_id: null, origin: "sdk" }))).toBe("—");
    expect(viaLabel(row({ session_id: null, origin: null }))).toBe("—");
  });

  it("maps a real origin into the same vocabulary as the session channels", () => {
    // Not `api`/`API` as two filter values for one channel.
    expect(viaLabel(row({ session_id: null, origin: "api" }))).toBe("API");
    expect(viaLabel(row({ session_id: null, origin: "mcp" }))).toBe("MCP");
    expect(viaLabel(row({ session_id: null, origin: "background" }))).toBe("Background");
  });

  it("gives a grouping caller undefined, not the rendered dash", () => {
    // Bucketing on viaLabel's dash filed every unidentified row under a
    // category literally named "—".
    expect(viaChannel(row({ session_id: null, origin: "sdk" }))).toBeUndefined();
    expect(viaChannel(row({ session_id: null, origin: null }))).toBeUndefined();
    expect(viaChannel(row({ session_id: "search-ui-1" }))).toBe("Cloud");
    expect(viaChannel(row({ session_id: null, origin: "api" }))).toBe("API");
  });

  it("names an OpenCode session, which had no channel entry at all", () => {
    expect(viaLabel(row({ session_id: "opencode_abc" }))).toBe("OpenCode");
  });

  it("names a dashboard upload Cloud, the same as a dashboard session", () => {
    // An upload/cognify is a pipeline run, not a conversation, so it carries
    // no session id — origin is the only thing that can say where it came
    // from, and it has to land on the same label the session path produces.
    expect(viaLabel(row({ session_id: null, origin: "cloud" }))).toBe("Cloud");
    expect(viaLabel(row({ session_id: "search-ui-1" }))).toBe("Cloud");
  });

  it("passes an origin it has not been taught through unchanged", () => {
    expect(viaLabel(row({ session_id: null, origin: "webhook" }))).toBe("webhook");
  });

  it("reads the client from the session list a pod running cognee PR 6191 sends", () => {
    expect(viaLabel(row({ session_ids: ["claude_1"], origin: "background" }))).toBe("Claude Code");
  });

  it("names the client when every session of a multi-session run agrees", () => {
    expect(viaLabel(row({ session_ids: ["claude_1", "claude_2"], origin: "background" }))).toBe("Claude Code");
  });

  it("falls back to origin when the sessions name different clients", () => {
    // An improve bridging a Claude Code session and a Cursor session has no
    // single client; claiming either would be wrong for the other.
    expect(viaLabel(row({ session_ids: ["claude_1", "cursor_2"], origin: "background" }))).toBe("Background");
    expect(viaChannel(row({ session_ids: ["claude_1", "cursor_2"], origin: "sdk" }))).toBeUndefined();
  });

  it("treats an empty list like no session", () => {
    expect(viaLabel(row({ session_ids: [], origin: "api" }))).toBe("API");
    expect(viaChannel(row({ session_ids: [], origin: null }))).toBeUndefined();
  });
});
