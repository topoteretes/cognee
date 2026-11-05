import { listSessions } from "@/modules/sessions/getSessions";
import type { CogneeInstance } from "@/modules/instances/types";

/**
 * The availability probe's promise is cached at module scope and shared by
 * every caller of listSessions. This pins the consequence: nothing
 * caller-specific may end up inside it.
 *
 * The regression it guards against was real. Threading one caller's AbortSignal
 * into the probe meant that when react-query cancelled that caller's query on
 * unmount — which it does on last-observer removal, i.e. navigating away
 * mid-probe — the shared promise rejected, the negative result was cached for
 * 30 seconds, and Search, Sessions, Integrations, onboarding detection, the
 * dashboard and getSessionStats all read "sessions unavailable" off one
 * navigation.
 */

let instanceSeq = 0;

/** A pod that honours an abort signal, so a leaked signal would be visible. */
function makeInstance(): { instance: CogneeInstance; calls: string[] } {
  const calls: string[] = [];
  const instance = {
    // A fresh id per test: the probe cache is keyed on it and persists across
    // tests in this file.
    instanceId: `probe-iso-${++instanceSeq}`,
    name: "test",
    fetch: (path: string, init?: RequestInit) => {
      calls.push(path);
      const signal = init?.signal;
      if (signal?.aborted) return Promise.reject(new DOMException("Aborted", "AbortError"));
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ sessions: [], total: 0, limit: 50, offset: 0, has_more: false }),
      } as unknown as Response);
    },
  } as unknown as CogneeInstance;
  return { instance, calls };
}

describe("sessions availability probe", () => {
  it("survives a caller whose request was already aborted", async () => {
    const { instance } = makeInstance();
    const aborted = new AbortController();
    aborted.abort();

    // The aborting caller's own list request fails, which is correct.
    await expect(listSessions(instance, { limit: 1 }, { signal: aborted.signal }))
      .rejects.toMatchObject({ name: "AbortError" });

    // The next caller must not inherit that. Before the fix it did, for 30s.
    const second = await listSessions(instance, { limit: 1 });
    expect(second.failed).toBeUndefined();
    expect(second.sessions).toEqual([]);
  });

  it("does not send a caller's signal on the probe request", async () => {
    const { instance, calls } = makeInstance();
    const aborted = new AbortController();
    aborted.abort();

    await expect(listSessions(instance, { limit: 1 }, { signal: aborted.signal }))
      .rejects.toMatchObject({ name: "AbortError" });

    // The probe ran and succeeded despite the caller being aborted, which is
    // only possible if the signal never reached it.
    expect(calls).toContain("/v1/sessions?limit=1");
    const probeSucceeded = await listSessions(instance, { limit: 1 });
    expect(probeSucceeded.failed).toBeUndefined();
  });

  it("caches one probe per instance rather than one per caller", async () => {
    const { instance, calls } = makeInstance();

    // limit=50, so the list requests are distinguishable from the probe's
    // limit=1 — with the same limit the two URLs are identical and the count
    // means nothing.
    await Promise.all([
      listSessions(instance, { limit: 50 }),
      listSessions(instance, { limit: 50 }),
      listSessions(instance, { limit: 50 }),
    ]);

    expect(calls.filter((c) => c === "/v1/sessions?limit=1")).toHaveLength(1);
    expect(calls.filter((c) => c === "/v1/sessions?limit=50")).toHaveLength(3);
  });

  it("reports a genuinely unreachable endpoint as failed, not as empty", async () => {
    // The other half of the fix: a failed probe must not read as "no sessions".
    const instance = {
      instanceId: `probe-iso-${++instanceSeq}`,
      name: "test",
      fetch: () => Promise.reject(new Error("pod down")),
    } as unknown as CogneeInstance;

    const page = await listSessions(instance, { limit: 1 });

    expect(page.failed).toBe(true);
    expect(page.sessions).toEqual([]);
  });
});
