import { listSearchSessions, SEARCH_SESSION_PREFIX } from "@/modules/sessions/getSessions";
import type { CogneeInstance } from "@/modules/instances/types";

const PAGE_SIZE = 200;

interface ListCall {
  range: string | null;
  limit: string | null;
  offset: string | null;
  timeoutMs: number | undefined;
}

const ME = "user-1";
const TEAMMATE = "user-2";

function sessionRow(spec: unknown) {
  // Anything that is not an id or an [id, userId] pair is served untouched, so
  // a page can carry a row that is not a row at all. Wrapping it in a
  // well-formed object instead made the malformed-row test unable to fail.
  if (typeof spec !== "string" && !Array.isArray(spec)) return spec;
  const [sessionId, userId] = Array.isArray(spec) ? spec : [spec, ME];
  return {
    session_id: sessionId,
    user_id: userId,
    dataset_id: null,
    status: "completed",
    effective_status: "completed",
    started_at: "2026-09-01T10:00:00.000Z",
    last_activity_at: "2026-09-01T10:05:00.000Z",
    ended_at: null,
    tokens_in: 0,
    tokens_out: 0,
    cost_usd: 0,
    error_count: 0,
    last_model: null,
  };
}

type Page = unknown[] | "fail";

let instanceCounter = 0;

/**
 * Stubs the pod at the HTTP boundary. `pages` is what each successive offset
 * returns; "fail" is a non-200, the failure listSessions swallows.
 */
function instanceServing(pages: Page[]): {
  instance: CogneeInstance;
  calls: ListCall[];
  probes: { timeoutMs: number | undefined }[];
} {
  const calls: ListCall[] = [];
  const probes: { timeoutMs: number | undefined }[] = [];
  const fetch = jest.fn((input: RequestInfo | URL, init?: RequestInit & { timeoutMs?: number }) => {
    const url = new URL(String(input), "https://pod.test");
    // The availability probe the module runs before its first real call.
    if (url.searchParams.get("limit") === "1") {
      probes.push({ timeoutMs: init?.timeoutMs });
      return Promise.resolve(new Response("{}", { status: 200 }));
    }
    calls.push({
      range: url.searchParams.get("range"),
      limit: url.searchParams.get("limit"),
      offset: url.searchParams.get("offset"),
      timeoutMs: init?.timeoutMs,
    });
    const index = Number(url.searchParams.get("offset") ?? 0) / PAGE_SIZE;
    const page = pages[index];
    if (page === "fail") return Promise.resolve(new Response("nope", { status: 502 }));
    const sessions = (page ?? []).map(sessionRow);
    return Promise.resolve(
      new Response(
        JSON.stringify({
          sessions,
          total: 0,
          limit: PAGE_SIZE,
          offset: index * PAGE_SIZE,
          has_more: index < pages.length - 1,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
  });
  // A distinct instanceId per test so the module's availability-probe cache,
  // which is keyed on it in production too, cannot leak between tests.
  instanceCounter += 1;
  const instance = {
    name: "CloudCognee",
    instanceId: `instance-${instanceCounter}`,
    fetch,
  } as unknown as CogneeInstance;
  return { instance, calls, probes };
}

const SEARCH_A = `${SEARCH_SESSION_PREFIX}1000`;
const SEARCH_B = `${SEARCH_SESSION_PREFIX}2000`;
const agentPage = (n = PAGE_SIZE) => Array.from({ length: n }, (_, i) => `claude_code_${i}`);

describe("listSearchSessions", () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it("returns only sessions started by the search page", async () => {
    const { instance } = instanceServing([[SEARCH_A, "claude_code_1", "cc_2", SEARCH_B]]);

    const { sessions } = await listSearchSessions(instance, 20, ME);

    expect(sessions.map((s) => s.session_id)).toEqual([SEARCH_A, SEARCH_B]);
  });

  it("keeps paging when the first page is entirely agent sessions", async () => {
    const { instance } = instanceServing([agentPage(), [SEARCH_A]]);

    const { sessions } = await listSearchSessions(instance, 20, ME);

    expect(sessions.map((s) => s.session_id)).toEqual([SEARCH_A]);
  });

  it("scans the whole history rather than a trailing time window", async () => {
    const { instance, calls } = instanceServing([[SEARCH_A]]);

    await listSearchSessions(instance, 20, ME);

    expect(calls[0].range).toBe("all");
  });

  it("allows the scan longer than the client's default GET timeout", async () => {
    const { instance, calls } = instanceServing([[SEARCH_A]]);

    await listSearchSessions(instance, 20, ME);

    expect(calls[0].timeoutMs).toBe(25_000);
  });

  // The opposite of the scan above, deliberately: the probe promise is cached
  // at module scope and shared by every caller, so a per-caller timeout would
  // have caller B waiting on caller A's budget. A busy pod failing the probe
  // is survivable because that now answers FAILED_PAGE, which the scan reports
  // as "couldn't load" rather than as an empty sidebar.
  it("gives the shared availability probe no per-caller timeout", async () => {
    const { instance, probes } = instanceServing([[SEARCH_A]]);

    await listSearchSessions(instance, 20, ME);

    expect(probes[0].timeoutMs).toBeUndefined();
  });

  it("stops scanning when the caller aborts between pages", async () => {
    const { instance, calls } = instanceServing([[SEARCH_B], [SEARCH_A]]);
    const controller = new AbortController();

    const scan = listSearchSessions(instance, 20, ME, controller.signal);
    controller.abort();
    const { complete } = await scan;

    expect(calls).toHaveLength(1);
    expect(complete).toBe(false);
  });

  it("reports a complete scan when the list is exhausted", async () => {
    const { instance } = instanceServing([[SEARCH_A]]);

    expect((await listSearchSessions(instance, 20, ME)).complete).toBe(true);
  });

  it("reports an incomplete scan when a page fails, rather than an empty list", async () => {
    const { instance } = instanceServing([agentPage(), "fail", [SEARCH_A]]);

    const result = await listSearchSessions(instance, 20, ME);

    expect(result.sessions).toEqual([]);
    expect(result.complete).toBe(false);
  });

  it("keeps what it already found when a later page fails", async () => {
    const { instance } = instanceServing([[SEARCH_A, ...agentPage(199)], "fail"]);

    const result = await listSearchSessions(instance, 20, ME);

    expect(result.sessions.map((s) => s.session_id)).toEqual([SEARCH_A]);
    expect(result.complete).toBe(false);
  });

  it("stops paging as soon as it has the conversations it was asked for", async () => {
    const { instance, calls } = instanceServing([[SEARCH_A, SEARCH_B], [`${SEARCH_SESSION_PREFIX}3`]]);

    const { sessions } = await listSearchSessions(instance, 2, ME);

    expect(sessions).toHaveLength(2);
    expect(calls).toHaveLength(1);
  });

  it("never returns more sessions than asked for", async () => {
    const many = Array.from({ length: 50 }, (_, i) => `${SEARCH_SESSION_PREFIX}${i}`);
    const { instance } = instanceServing([many]);

    const { sessions } = await listSearchSessions(instance, 20, ME);

    expect(sessions).toHaveLength(20);
  });

  it("gives up after a bounded number of pages and admits the scan is partial", async () => {
    const { instance, calls } = instanceServing(Array.from({ length: 7 }, () => agentPage()));

    const result = await listSearchSessions(instance, 20, ME);

    expect(result.sessions).toEqual([]);
    expect(result.complete).toBe(false);
    expect(calls).toHaveLength(5);
  });

  it("leaves a teammate's search session out, even though the pod returns it", async () => {
    const { instance } = instanceServing([[[SEARCH_A, TEAMMATE], [SEARCH_B, ME]]]);

    const { sessions } = await listSearchSessions(instance, 20, ME);

    expect(sessions.map((s) => s.session_id)).toEqual([SEARCH_B]);
  });

  it("survives a 200 whose rows carry no usable session id", async () => {
    const { instance } = instanceServing([[null, 42, { nested: true }, SEARCH_A]]);

    const { sessions } = await listSearchSessions(instance, 20, ME);

    expect(sessions.map((s) => s.session_id)).toEqual([SEARCH_A]);
  });
});
