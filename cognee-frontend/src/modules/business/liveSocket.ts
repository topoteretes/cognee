import type { SessionEvent } from "./types";
import { filterSessionEvents } from "./sessionEvent";

// Server contract (fixed, already deployed — see WS
// /api/v1/visualize/subscribe/{dataset_id}): no client frame is ever read by
// the server, so this file only ever receives.
export type LiveSocketFrame =
  | { kind: "ready"; dataset_id: string; cursor: string | null }
  // `cursor` is null when the server sent one this client cannot use. That is
  // "do not advance the resume point", never "start over" — see the single
  // cursor policy documented in parseFrame.
  | { kind: "live_events"; events: SessionEvent[]; cursor: string | null }
  | { kind: "graph_grew"; pipeline_run_id: string }
  | { kind: "heartbeat"; time: string };

export interface LiveSocketHandlers {
  onFrame: (frame: LiveSocketFrame) => void;
  // Fires once when the fast reconnect ladder is exhausted, so the caller can
  // fall back to the HTTP poll and alert. "rejected" (close 1008 —
  // unauthenticated/no read permission) is terminal, since a retry replays
  // the same rejection. "unreachable" is 5 consecutive attempts that never
  // completed a handshake, and is NOT terminal: the socket keeps trying on a
  // slow timer afterwards, so a pod restart or a cold-start wake recovers on
  // its own instead of leaving the tab downgraded until the user navigates.
  // "refused" is that same ladder exhausted by the browser rejecting the URL
  // outright (`new WebSocket` throwing), kept apart from "unreachable"
  // because the remedies share nothing: a refused URL is ours to fix (scheme,
  // CSP, a malformed id), an unreachable pod is not. The reason is the only
  // thing ever reported, so splitting it leaks nothing.
  onGiveUp: (reason: "rejected" | "unreachable" | "refused") => void;
  // Fires the moment a reconnect attempt starts, so the caller can mirror the
  // poll transport's "failed tick -> live:false" until the next `ready`.
  onReconnecting: () => void;
  // Called when a `live_events` frame carried entries the guard could not
  // accept. The frame itself is still delivered (see parseFrame) — this is
  // the signal that would otherwise not exist.
  onDroppedEvents?: (dropped: number, total: number) => void;
}

export interface LiveSocketHandle {
  close: () => void;
}

type SocketTarget = (path: string) => { url: string; apiKey: string };

const RECONNECT_DELAYS_MS = [1000, 2000, 4000, 8000];
export const MAX_FAILURES = 5;
// The whole ladder is ~15s, which is shorter than both a pod restart and the
// ~60s cold-start wake of a suspended workspace (CLO-713). So once it is
// exhausted the socket keeps retrying at this interval instead of staying
// dead: the caller has already fallen back to the poll and alerted by then,
// and a recovered socket announces itself with a fresh `ready`.
export const UNREACHABLE_RETRY_MS = 5 * 60_000;
// How long total silence means "dead" rather than "busy".
//
// The server's heartbeat is NOT on a fixed clock. It sends one every 15 ticks,
// and each tick is a 1s sleep PLUS that tick's DB work, so the real spacing is
// 15 * (1s + work) — it stretches with pod load rather than staying at the
// nominal 15s:
//
//     work/tick   0.4s -> ~21s      2s -> ~45s      10s -> ~165s
//
// Any fixed window is therefore a bet on how slow the pod gets, and the load
// that stretches the spacing is exactly the load this feature exists for:
// getBrainsSummary.ts documents this same pod going from ~400ms idle to over
// 10s during a cognify run. 45s lost that bet at 2s of per-tick work, and 60s
// loses it at 3s, so both turn a busy pod into a reconnect loop.
//
// 180s is sized to cover the 10s-per-tick case the latency notes actually
// measured. The cost of being generous is bounded and small: a genuinely dead
// socket usually arrives as an `onclose` long before this fires (that is the
// common path), so this only backstops the "connected but silent" case, and
// while it waits the caller is still on a proven socket. The cost of being
// tight is a reconnect on every busy pod, which is worse.
export const HEARTBEAT_WATCHDOG_MS = 180_000;
const WATCHDOG_CLOSE_CODE = 4000;
const REJECTED_CLOSE_CODE = 1008;

// `datasetId` is the dataset this socket was opened for. `ready` carries the
// server's own idea of that, and validating a field then ignoring it is how a
// mismatch becomes invisible: a `ready` for the wrong dataset would switch off
// the graph poll for THIS dataset while `graph_grew` invalidates the locally
// known id. Unreachable today (one dataset per socket), which is exactly why
// it is cheap to pin now, before the pod ever multiplexes.
function parseFrame(
  raw: string,
  datasetId: string,
  onDroppedEvents?: (dropped: number, total: number) => void,
): LiveSocketFrame | null {
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return null;
  }
  if (typeof data !== "object" || data === null || !("kind" in data)) return null;
  const kind = (data as { kind: unknown }).kind;

  switch (kind) {
    case "ready": {
      const { dataset_id, cursor } = data as { dataset_id?: unknown; cursor?: unknown };
      if (typeof dataset_id !== "string" || dataset_id !== datasetId) return null;
      if (cursor !== null && typeof cursor !== "string") return null;
      return { kind: "ready", dataset_id, cursor: cursor ?? null };
    }
    case "live_events": {
      const { events, cursor } = data as { events?: unknown; cursor?: unknown };
      if (!Array.isArray(events)) return null;
      // Unusable entries are dropped, the frame is not. Rejecting the whole
      // frame would also throw away its cursor, so the next reconnect would
      // resume from before it and replay everything it did contain - one bad
      // entry would stall the stream rather than cost one event.
      //
      // The cursor follows one policy shared with the HTTP poll
      // (getLiveEvents/pollLiveEvents): an unusable cursor NEVER advances the
      // resume point and NEVER resets it. Rejecting the frame over it would
      // drop real events; resetting to null would make the next resume replay
      // the whole history. Keeping the last good value costs at most a small
      // replay on reconnect, which the dedupe absorbs.
      return {
        kind: "live_events",
        events: filterSessionEvents(events, onDroppedEvents),
        cursor: typeof cursor === "string" ? cursor : null,
      };
    }
    case "graph_grew": {
      const { pipeline_run_id } = data as { pipeline_run_id?: unknown };
      if (typeof pipeline_run_id !== "string") return null;
      return { kind: "graph_grew", pipeline_run_id };
    }
    case "heartbeat": {
      const { time } = data as { time?: unknown };
      if (typeof time !== "string") return null;
      return { kind: "heartbeat", time };
    }
    default:
      return null;
  }
}

function buildUrl(socketTarget: SocketTarget, datasetId: string, since: string | null): string {
  // Encoded even though every caller passes a UUID today: this id reaches
  // here from a graph payload, and an unencoded `/` or `?` in a path segment
  // would silently re-route the request or graft a query param onto it while
  // the token still lands at the end.
  const { url, apiKey } = socketTarget(`/v1/visualize/subscribe/${encodeURIComponent(datasetId)}`);
  const wsUrl = new URL(url);
  wsUrl.searchParams.set("token", apiKey);
  if (since) wsUrl.searchParams.set("since", since);
  return wsUrl.toString();
}

// Opens (and transparently reconnects) the Mindmap live-updates socket.
// Owns its own resume cursor across reconnects — the caller only supplies the
// starting point once — and a failure counter that resets only on a `ready`
// frame, since that's the completed handshake, not just a completed TCP
// connect (a socket that connects and is immediately dropped never proved it
// could actually authenticate).
export function openLiveSocket(
  socketTarget: SocketTarget,
  datasetId: string,
  since: string | null,
  handlers: LiveSocketHandlers,
): LiveSocketHandle {
  let closed = false;
  let socket: WebSocket | null = null;
  let watchdogTimer: ReturnType<typeof setTimeout> | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let failures = 0;
  let resumeCursor = since;
  // Set once the fast ladder is exhausted: the caller has been told and has
  // fallen back, so later failures re-arm the slow timer silently instead of
  // re-alerting on every round.
  let gaveUp = false;
  // Whether the most recent failure was the browser refusing to construct the
  // socket at all, rather than a connection that opened and then died.
  let lastFailureWasRefusal = false;

  function armWatchdog(): void {
    if (watchdogTimer) clearTimeout(watchdogTimer);
    watchdogTimer = setTimeout(() => socket?.close(WATCHDOG_CLOSE_CODE), HEARTBEAT_WATCHDOG_MS);
  }

  function scheduleReconnect(): void {
    if (closed) return;
    failures++;
    // Already given up once: keep trying, quietly, on the slow timer. The
    // caller is on the poll and stays there until a `ready` arrives.
    if (gaveUp) {
      reconnectTimer = setTimeout(connect, UNREACHABLE_RETRY_MS);
      return;
    }
    if (failures >= MAX_FAILURES) {
      gaveUp = true;
      handlers.onGiveUp(lastFailureWasRefusal ? "refused" : "unreachable");
      reconnectTimer = setTimeout(connect, UNREACHABLE_RETRY_MS);
      return;
    }
    handlers.onReconnecting();
    const delay = RECONNECT_DELAYS_MS[Math.min(failures - 1, RECONNECT_DELAYS_MS.length - 1)];
    reconnectTimer = setTimeout(connect, delay);
  }

  function handleMessage(raw: string): void {
    if (closed) return;
    const frame = parseFrame(raw, datasetId, handlers.onDroppedEvents);
    if (!frame) return;
    armWatchdog();
    if (frame.kind === "ready") {
      failures = 0;
      // A slow retry that finally lands puts the socket fully back in
      // service, including its fast ladder for the NEXT outage.
      gaveUp = false;
      resumeCursor = frame.cursor ?? resumeCursor;
    } else if (frame.kind === "live_events" && frame.cursor !== null) {
      resumeCursor = frame.cursor;
    }
    handlers.onFrame(frame);
  }

  function handleClose(code: number): void {
    if (closed) return;
    if (watchdogTimer) clearTimeout(watchdogTimer);
    if (code === REJECTED_CLOSE_CODE) {
      closed = true;
      handlers.onGiveUp("rejected");
      return;
    }
    scheduleReconnect();
  }

  function connect(): void {
    if (closed) return;
    let ws: WebSocket;
    try {
      ws = new WebSocket(buildUrl(socketTarget, datasetId, resumeCursor));
      lastFailureWasRefusal = false;
    } catch {
      lastFailureWasRefusal = true;
      // `new WebSocket` throws synchronously for a URL the browser refuses
      // outright (Chrome on a `ws://` URL from an https page, which
      // createPodClient's `base.replace(/^http/, "ws")` produces for any
      // non-https pod origin). Unguarded, that exception escaped this
      // function into the caller's useEffect and error-boundaried the whole
      // Mindmap instead of degrading to the poll — the exact failure this
      // fallback exists for. Deliberately swallowed rather than rethrown or
      // logged with the error: the browser puts the full URL in the message
      // and that URL carries the tenant API key as `?token=`, so reporting
      // it anywhere would leak the credential. The give-up path below is
      // already the loud, key-free signal for this.
      scheduleReconnect();
      return;
    }
    socket = ws;
    // Armed from connect time, not from the first received frame — a
    // connection that never sends anything at all (not even the mandatory
    // first `ready`) is exactly the "dead, not quiet" case this watchdog
    // exists to catch (see HEARTBEAT_WATCHDOG_MS for how the window is sized).
    armWatchdog();
    ws.onmessage = (event) => {
      if (typeof event.data === "string") handleMessage(event.data);
    };
    ws.onclose = (event) => handleClose(event.code);
  }

  connect();

  return {
    close: () => {
      closed = true;
      if (watchdogTimer) clearTimeout(watchdogTimer);
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (socket) {
        socket.onmessage = null;
        socket.onclose = null;
        socket.close();
      }
    },
  };
}
