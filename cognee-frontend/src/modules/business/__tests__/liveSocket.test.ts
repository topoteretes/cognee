import {
  openLiveSocket,
  HEARTBEAT_WATCHDOG_MS,
  UNREACHABLE_RETRY_MS,
  MAX_FAILURES,
  type LiveSocketFrame,
  type LiveSocketHandlers,
} from "../liveSocket";

// Minimal WebSocket double: captures the constructed URL and lets a test
// drive onmessage/onclose directly, without a real network socket.
class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  url: string;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: ((event: { code: number }) => void) | null = null;
  closeCodes: number[] = [];

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  close(code = 1000): void {
    this.closeCodes.push(code);
    this.onclose?.({ code });
  }

  send(data: string): void {
    this.onmessage?.({ data });
  }
}

function socketTarget(path: string): { url: string; apiKey: string } {
  return { url: `wss://tenant-x.example.com/api${path}`, apiKey: "secret-key" };
}

function handlers(): LiveSocketHandlers & {
  frames: LiveSocketFrame[];
  giveUps: Array<"rejected" | "unreachable" | "refused">;
  reconnects: number;
  drops: Array<[number, number]>;
} {
  const frames: LiveSocketFrame[] = [];
  const giveUps: Array<"rejected" | "unreachable" | "refused"> = [];
  const drops: Array<[number, number]> = [];
  let reconnects = 0;
  return {
    frames,
    giveUps,
    drops,
    get reconnects() {
      return reconnects;
    },
    onFrame: (frame) => frames.push(frame),
    onGiveUp: (reason) => giveUps.push(reason),
    onReconnecting: () => {
      reconnects++;
    },
    onDroppedEvents: (dropped, total) => drops.push([dropped, total]),
  };
}

function latestSocket(): FakeWebSocket {
  const socket = FakeWebSocket.instances.at(-1);
  if (!socket) throw new Error("no FakeWebSocket instance was constructed");
  return socket;
}

beforeEach(() => {
  jest.useFakeTimers();
  FakeWebSocket.instances = [];
  (global as unknown as { WebSocket: unknown }).WebSocket = FakeWebSocket;
});

afterEach(() => {
  jest.useRealTimers();
});

describe("openLiveSocket — handshake URL", () => {
  it("carries the dataset id and apiKey as a token query param", () => {
    openLiveSocket(socketTarget, "ds-1", null, handlers());
    const url = new URL(latestSocket().url);
    expect(url.pathname).toBe("/api/v1/visualize/subscribe/ds-1");
    expect(url.searchParams.get("token")).toBe("secret-key");
  });

  it("omits `since` when no cursor is given", () => {
    openLiveSocket(socketTarget, "ds-1", null, handlers());
    expect(new URL(latestSocket().url).searchParams.has("since")).toBe(false);
  });

  it("carries `since` as the reconnect cursor when one is given", () => {
    openLiveSocket(socketTarget, "ds-1", "2026-08-20T06:12:17Z", handlers());
    expect(new URL(latestSocket().url).searchParams.get("since")).toBe("2026-08-20T06:12:17Z");
  });
});

describe("openLiveSocket — frames", () => {
  it("delivers a ready frame", () => {
    const h = handlers();
    openLiveSocket(socketTarget, "ds-1", null, h);
    latestSocket().send(JSON.stringify({ kind: "ready", dataset_id: "ds-1", cursor: "c0" }));
    expect(h.frames).toEqual([{ kind: "ready", dataset_id: "ds-1", cursor: "c0" }]);
  });

  it("delivers a live_events frame", () => {
    const h = handlers();
    openLiveSocket(socketTarget, "ds-1", null, h);
    const events = [{ kind: "search", qa_id: "q1" }];
    latestSocket().send(JSON.stringify({ kind: "live_events", events, cursor: "c1" }));
    expect(h.frames).toEqual([{ kind: "live_events", events, cursor: "c1" }]);
  });

  it("delivers a graph_grew frame", () => {
    const h = handlers();
    openLiveSocket(socketTarget, "ds-1", null, h);
    latestSocket().send(JSON.stringify({ kind: "graph_grew", pipeline_run_id: "run-1" }));
    expect(h.frames).toEqual([{ kind: "graph_grew", pipeline_run_id: "run-1" }]);
  });

  it("delivers a heartbeat frame", () => {
    const h = handlers();
    openLiveSocket(socketTarget, "ds-1", null, h);
    latestSocket().send(JSON.stringify({ kind: "heartbeat", time: "2026-08-20T06:12:17Z" }));
    expect(h.frames).toEqual([{ kind: "heartbeat", time: "2026-08-20T06:12:17Z" }]);
  });

  it("ignores a frame with an unknown kind", () => {
    const h = handlers();
    openLiveSocket(socketTarget, "ds-1", null, h);
    latestSocket().send(JSON.stringify({ kind: "something_else" }));
    expect(h.frames).toEqual([]);
  });

  it("ignores a frame missing required fields", () => {
    const h = handlers();
    openLiveSocket(socketTarget, "ds-1", null, h);
    latestSocket().send(JSON.stringify({ kind: "live_events" }));
    expect(h.frames).toEqual([]);
  });

  it("ignores a non-JSON payload instead of throwing", () => {
    const h = handlers();
    openLiveSocket(socketTarget, "ds-1", null, h);
    expect(() => latestSocket().send("not json")).not.toThrow();
    expect(h.frames).toEqual([]);
  });

  it("accepts an event identified by time alone", () => {
    const h = handlers();
    openLiveSocket(socketTarget, "ds-1", null, h);
    const events = [{ kind: "improve", time: "t1" }];
    latestSocket().send(JSON.stringify({ kind: "live_events", events, cursor: "c1" }));
    expect(h.frames).toEqual([{ kind: "live_events", events, cursor: "c1" }]);
  });

  it("drops an event carrying no identity, since the dedupe key cannot tell two of them apart", () => {
    const h = handlers();
    openLiveSocket(socketTarget, "ds-1", null, h);
    const good = { kind: "search", qa_id: "q1", time: "t1" };
    latestSocket().send(JSON.stringify({ kind: "live_events", events: [{ kind: "search" }, good], cursor: "c1" }));
    expect(h.frames).toEqual([{ kind: "live_events", events: [good], cursor: "c1" }]);
  });

  it("drops an event whose known fields have the wrong type", () => {
    const h = handlers();
    openLiveSocket(socketTarget, "ds-1", null, h);
    const events = [
      { qa_id: 7, time: "t1" },
      { qa_id: "q1", time: "t1", node_ids: ["n1", 2] },
    ];
    latestSocket().send(JSON.stringify({ kind: "live_events", events, cursor: "c1" }));
    expect(h.frames).toEqual([{ kind: "live_events", events: [], cursor: "c1" }]);
  });

  it("keeps the frame's cursor even when every event in it was dropped", () => {
    const h = handlers();
    openLiveSocket(socketTarget, "ds-1", null, h);
    // Rejecting the frame outright would resume the next connection from
    // before this cursor and replay whatever it did carry.
    latestSocket().send(JSON.stringify({ kind: "live_events", events: [{}], cursor: "c1" }));
    latestSocket().close(1006);
    jest.advanceTimersByTime(1000);
    expect(new URL(latestSocket().url).searchParams.get("since")).toBe("c1");
  });
});

describe("openLiveSocket — watchdog", () => {
  const WATCHDOG_CLOSE_CODE = 4000;

  it("closes with 4000 after the watchdog window of total silence", () => {
    openLiveSocket(socketTarget, "ds-1", null, handlers());
    jest.advanceTimersByTime(HEARTBEAT_WATCHDOG_MS);
    expect(latestSocket().closeCodes).toContain(WATCHDOG_CLOSE_CODE);
  });

  it("a periodic heartbeat keeps the connection alive past the watchdog window", () => {
    openLiveSocket(socketTarget, "ds-1", null, handlers());
    for (let i = 0; i < 3; i++) {
      jest.advanceTimersByTime(20_000);
      latestSocket().send(JSON.stringify({ kind: "heartbeat", time: "t" }));
    }
    expect(latestSocket().closeCodes).toEqual([]);
  });
});

describe("openLiveSocket — reconnect and give-up", () => {
  it("backs off 1s/2s/4s/8s across consecutive failed connects", () => {
    openLiveSocket(socketTarget, "ds-1", null, handlers());
    latestSocket().close(1006);
    expect(FakeWebSocket.instances).toHaveLength(1);

    jest.advanceTimersByTime(999);
    expect(FakeWebSocket.instances).toHaveLength(1);
    jest.advanceTimersByTime(1);
    expect(FakeWebSocket.instances).toHaveLength(2);

    latestSocket().close(1006);
    jest.advanceTimersByTime(1999);
    expect(FakeWebSocket.instances).toHaveLength(2);
    jest.advanceTimersByTime(1);
    expect(FakeWebSocket.instances).toHaveLength(3);
  });

  it("resets the failure counter only on a ready frame, not on a bare reconnect", () => {
    openLiveSocket(socketTarget, "ds-1", null, handlers());
    latestSocket().close(1006); // failure #1
    jest.advanceTimersByTime(1000); // 1s backoff -> attempt #2

    latestSocket().send(JSON.stringify({ kind: "ready", dataset_id: "ds-1", cursor: null }));
    latestSocket().close(1006); // failure count must restart from 0 here

    // If the reset had NOT happened this would need a 2s wait (failure #2).
    jest.advanceTimersByTime(1000);
    expect(FakeWebSocket.instances).toHaveLength(3);
  });

  it("gives up as unreachable after 5 consecutive failures, with no further reconnect", () => {
    const h = handlers();
    openLiveSocket(socketTarget, "ds-1", null, h);
    const delays = [1000, 2000, 4000, 8000];
    for (const delay of delays) {
      latestSocket().close(1006);
      jest.advanceTimersByTime(delay);
    }
    expect(FakeWebSocket.instances).toHaveLength(5);

    latestSocket().close(1006); // 5th failure
    expect(h.giveUps).toEqual(["unreachable"]);

    jest.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances).toHaveLength(5);
  });

  it("gives up as rejected on a 1008 close and never reconnects", () => {
    const h = handlers();
    openLiveSocket(socketTarget, "ds-1", null, h);
    latestSocket().close(1008);

    expect(h.giveUps).toEqual(["rejected"]);
    jest.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("reconnects with `since` set to the last cursor the socket reported", () => {
    openLiveSocket(socketTarget, "ds-1", null, handlers());
    latestSocket().send(JSON.stringify({ kind: "ready", dataset_id: "ds-1", cursor: "c0" }));
    latestSocket().send(JSON.stringify({ kind: "live_events", events: [], cursor: "c1" }));
    latestSocket().close(1006);
    jest.advanceTimersByTime(1000);

    expect(new URL(latestSocket().url).searchParams.get("since")).toBe("c1");
  });
});

describe("openLiveSocket — close()", () => {
  it("silences a frame delivered after close() is called", () => {
    const h = handlers();
    const handle = openLiveSocket(socketTarget, "ds-1", null, h);
    const socket = latestSocket();
    handle.close();
    socket.send(JSON.stringify({ kind: "ready", dataset_id: "ds-1", cursor: "c0" }));
    expect(h.frames).toEqual([]);
  });

  it("cancels a pending reconnect", () => {
    const handle = openLiveSocket(socketTarget, "ds-1", null, handlers());
    latestSocket().close(1006);
    handle.close();
    jest.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });
});

describe("openLiveSocket — a WebSocket the browser refuses to construct", () => {
  // Chrome throws synchronously from `new WebSocket` for a URL it rejects
  // outright, which createPodClient produces for any non-https pod origin
  // (`base.replace(/^http/, "ws")` yields `ws://` on an https page).
  function throwOnConstruct(times: number): void {
    let constructed = 0;
    class RefusingWebSocket extends FakeWebSocket {
      constructor(url: string) {
        super(url);
        constructed++;
        if (constructed <= times) throw new Error(`refused ${url}`);
      }
    }
    (globalThis as unknown as { WebSocket: unknown }).WebSocket = RefusingWebSocket;
  }

  it("does not let the exception escape openLiveSocket", () => {
    throwOnConstruct(1);
    const h = handlers();
    expect(() => openLiveSocket(socketTarget, "ds-1", null, h)).not.toThrow();
  });

  it("treats it as a failed attempt and reconnects instead of dying", () => {
    throwOnConstruct(1);
    const h = handlers();
    openLiveSocket(socketTarget, "ds-1", null, h);
    expect(h.reconnects).toBe(1);
    jest.advanceTimersByTime(1000);
    // Second construct succeeds, so a `ready` can still arrive.
    latestSocket().send(JSON.stringify({ kind: "ready", dataset_id: "ds-1", cursor: null }));
    expect(h.frames.map((f) => f.kind)).toContain("ready");
  });

  it("gives up loudly rather than silently, when every construct is refused", () => {
    throwOnConstruct(Number.MAX_SAFE_INTEGER);
    const h = handlers();
    openLiveSocket(socketTarget, "ds-1", null, h);
    jest.advanceTimersByTime(1000 + 2000 + 4000 + 8000);
    // "refused", not "unreachable": the browser rejected the URL, the pod was
    // never reached, and the two need different remedies.
    expect(h.giveUps).toEqual(["refused"]);
  });
});

describe("openLiveSocket — an exhausted ladder is not the end", () => {
  function exhaustLadder(h: ReturnType<typeof handlers>): void {
    openLiveSocket(socketTarget, "ds-1", null, h);
    for (let i = 0; i < MAX_FAILURES; i++) {
      latestSocket().close(1006);
      jest.advanceTimersByTime(8000);
    }
  }

  it("reports unreachable exactly once, however long the outage lasts", () => {
    const h = handlers();
    exhaustLadder(h);
    expect(h.giveUps).toEqual(["unreachable"]);
    for (let i = 0; i < 3; i++) {
      jest.advanceTimersByTime(UNREACHABLE_RETRY_MS);
      latestSocket().close(1006);
    }
    // Still one — the caller is already on the poll and does not need an
    // alert per retry round.
    expect(h.giveUps).toEqual(["unreachable"]);
  });

  it("keeps retrying on the slow timer, so a pod restart recovers on its own", () => {
    const h = handlers();
    exhaustLadder(h);
    const before = FakeWebSocket.instances.length;
    jest.advanceTimersByTime(UNREACHABLE_RETRY_MS);
    expect(FakeWebSocket.instances.length).toBe(before + 1);
    latestSocket().send(JSON.stringify({ kind: "ready", dataset_id: "ds-1", cursor: null }));
    expect(h.frames.at(-1)).toEqual({ kind: "ready", dataset_id: "ds-1", cursor: null });
  });

  it("restores the fast ladder once a slow retry lands, for the next outage", () => {
    const h = handlers();
    exhaustLadder(h);
    jest.advanceTimersByTime(UNREACHABLE_RETRY_MS);
    latestSocket().send(JSON.stringify({ kind: "ready", dataset_id: "ds-1", cursor: null }));
    const reconnectsBefore = h.reconnects;
    // A recovered socket dropping again must go back to the 1s ladder, not
    // straight to another 5-minute wait.
    const before = FakeWebSocket.instances.length;
    latestSocket().close(1006);
    expect(h.reconnects).toBe(reconnectsBefore + 1);
    jest.advanceTimersByTime(1000);
    expect(FakeWebSocket.instances.length).toBe(before + 1);
  });

  it("stops retrying for good once close() is called", () => {
    const h = handlers();
    const handle = openLiveSocket(socketTarget, "ds-1", null, h);
    for (let i = 0; i < MAX_FAILURES; i++) {
      latestSocket().close(1006);
      jest.advanceTimersByTime(8000);
    }
    handle.close();
    const before = FakeWebSocket.instances.length;
    jest.advanceTimersByTime(UNREACHABLE_RETRY_MS * 3);
    expect(FakeWebSocket.instances.length).toBe(before);
  });
});

describe("openLiveSocket — frames that are not this socket's business", () => {
  it("ignores a ready for a different dataset", () => {
    const h = handlers();
    openLiveSocket(socketTarget, "ds-1", null, h);
    latestSocket().send(JSON.stringify({ kind: "ready", dataset_id: "ds-other", cursor: "c1" }));
    // Dropped whole: this frame would otherwise switch the caller's graph
    // poll off for ds-1 on the strength of another dataset's handshake.
    expect(h.frames).toEqual([]);
  });

  it("encodes the dataset id, so a stray separator cannot re-route the request", () => {
    const h = handlers();
    // Unreachable with today's UUIDs, but this id arrives from a graph
    // payload: an unencoded "/" would move the request to another path and a
    // "?" would graft a query param on while `token` still lands at the end.
    openLiveSocket(socketTarget, "a/b?c=1", null, h);
    const url = new URL(latestSocket().url);
    expect(url.pathname).toBe("/api/v1/visualize/subscribe/a%2Fb%3Fc%3D1");
    expect([...url.searchParams.keys()]).toEqual(["token"]);
  });

  it("reports how many live events it had to drop", () => {
    const h = handlers();
    openLiveSocket(socketTarget, "ds-1", null, h);
    latestSocket().send(
      JSON.stringify({
        kind: "live_events",
        cursor: "c1",
        events: [
          { kind: "search", qa_id: "q1", time: "2026-01-01T00:00:00Z" },
          { kind: "search", question: "no identity field" },
          { kind: "search", qa_id: "q2", time: 1735689600000 },
        ],
      }),
    );
    expect(h.drops).toEqual([[2, 3]]);
    expect(h.frames).toEqual([
      { kind: "live_events", cursor: "c1", events: [{ kind: "search", qa_id: "q1", time: "2026-01-01T00:00:00Z" }] },
    ]);
  });
});

describe("openLiveSocket — the timing constants themselves", () => {
  // Importing the constant and advancing by it makes the watchdog test say
  // "it fires after however long it is set to", which is true of any value.
  // The window is a tuning decision against a server heartbeat whose spacing
  // scales with pod load (see HEARTBEAT_WATCHDOG_MS), so the literal is the
  // thing that needs pinning: a change to it should require reading that
  // reasoning, not slip through green.
  it("waits 180s for total silence, sized for a 10s-per-tick pod", () => {
    expect(HEARTBEAT_WATCHDOG_MS).toBe(180_000);
  });

  it("does not fire one tick early", () => {
    openLiveSocket(socketTarget, "ds-1", null, handlers());
    jest.advanceTimersByTime(HEARTBEAT_WATCHDOG_MS - 1);
    expect(latestSocket().closeCodes).toEqual([]);
    jest.advanceTimersByTime(1);
    // 4000 is this client's own "I gave up waiting" code, not a server one.
    expect(latestSocket().closeCodes).toContain(4000);
  });

  it("retries a given-up socket every 5 minutes, not on some other clock", () => {
    expect(UNREACHABLE_RETRY_MS).toBe(300_000);
  });

  it("clears a pending reconnect timer on close, so nothing fires after unmount", () => {
    const h = handlers();
    const handle = openLiveSocket(socketTarget, "ds-1", null, h);
    latestSocket().close(1006);
    // A reconnect is now scheduled 1s out. close() has to cancel the timer
    // itself: relying on connect()'s early return leaves a live timeout
    // holding this closure alive, and up to 5 minutes of it once the slow
    // retry is the one pending.
    handle.close();
    // Measured immediately, not after advancing: letting the clock run would
    // fire the timer, connect() would early-return on `closed`, and the count
    // would read 0 whether or not close() ever cancelled anything.
    expect(jest.getTimerCount()).toBe(0);
    const before = FakeWebSocket.instances.length;
    jest.advanceTimersByTime(UNREACHABLE_RETRY_MS * 2);
    expect(FakeWebSocket.instances.length).toBe(before);
  });
});
