/**
 * Server-sent events read off a `fetch` response.
 *
 * `EventSource` only speaks GET, and the pod streams an answer from the same
 * POST whose body carries the query, so the frames are decoded from
 * `Response.body` here instead. Dispatch follows the SSE spec: fields
 * accumulate until a blank line, `data:` lines join with newlines, and comment
 * lines (a leading `:`) are dropped — those are the keepalives the pod sends
 * while retrieval and persistence run without producing tokens. A network
 * chunk can end anywhere, mid-line included, so nothing is dispatched before
 * its terminating blank line has arrived.
 */
import { HttpError } from "./errors";

export interface SseFrame {
  event: string;
  data: string;
}

export const SSE_MEDIA_TYPE = "text/event-stream";
// JSON is still acceptable, ranked lower, so a pod that cannot stream the
// request answers with the plain payload instead of a 406.
export const SSE_ACCEPT = `${SSE_MEDIA_TYPE}, application/json;q=0.9`;
// What the pod's own catch-all maps an unexpected failure to, for an error
// frame that carries no status of its own.
const ERROR_FRAME_FALLBACK_STATUS = 409;

/** Whether the pod answered with a stream, rather than the JSON fallback. */
export function isEventStream(response: Response): boolean {
  return (response.headers.get("content-type") ?? "").toLowerCase().startsWith(SSE_MEDIA_TYPE);
}

/** A frame's JSON object, or `{}` for data that is not one. */
export function parseFrameData(data: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(data);
    return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}

/**
 * A failure reported inside a 200: once the first frame is out the status
 * line is fixed, so the pod carries the status the JSON path would have used
 * in an `error` frame. Rebuilt as the same HttpError the JSON path throws.
 */
export function errorFrameToHttpError(payload: Record<string, unknown>, fallbackMessage: string): HttpError {
  const status = typeof payload.status === "number" ? payload.status : ERROR_FRAME_FALLBACK_STATUS;
  const message = typeof payload.message === "string" ? payload.message : fallbackMessage;
  return new HttpError(status, "", message, { error: message });
}

const DEFAULT_EVENT = "message";

class FrameAssembler {
  private event = DEFAULT_EVENT;
  private data: string[] = [];

  /** Feed one line without its terminator; returns a frame on a blank line. */
  line(raw: string): SseFrame | null {
    const line = raw.endsWith("\r") ? raw.slice(0, -1) : raw;
    if (line === "") return this.flush();
    if (line.startsWith(":")) return null;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    const rawValue = colon === -1 ? "" : line.slice(colon + 1);
    const value = rawValue.startsWith(" ") ? rawValue.slice(1) : rawValue;
    if (field === "event") this.event = value;
    else if (field === "data") this.data.push(value);
    return null;
  }

  /** The pending frame, if any data was collected; resets either way. */
  flush(): SseFrame | null {
    const frame = this.data.length > 0 ? { event: this.event, data: this.data.join("\n") } : null;
    this.event = DEFAULT_EVENT;
    this.data = [];
    return frame;
  }
}

/**
 * Yields frames as they complete. Leaving the loop early cancels the body so
 * the connection is released rather than held until garbage collection.
 *
 * `idleTimeoutMs` puts a ceiling the caller controls on the silence between
 * reads. Every read that resolves re-arms it, keepalive comments included,
 * which is what they are sent for, so a slow stream that keeps talking is not
 * cut and a stalled one is. When it fires the body is cancelled and the
 * generator throws, rather than ending as if the server had closed the stream,
 * which a caller waiting for a final frame would report as a stream that
 * stopped short.
 */
export async function* readSseFrames(
  body: ReadableStream<Uint8Array>,
  idleTimeoutMs?: number,
): AsyncGenerator<SseFrame> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const assembler = new FrameAssembler();
  let buffer = "";
  let exhausted = false;
  let quiet: ReturnType<typeof setTimeout> | undefined;
  let wentQuiet = false;
  const armIdleTimer = (): void => {
    if (idleTimeoutMs === undefined) return;
    quiet = setTimeout(() => {
      wentQuiet = true;
      void reader.cancel().catch(() => undefined);
    }, idleTimeoutMs);
  };
  try {
    while (!exhausted) {
      armIdleTimer();
      const { value, done } = await reader.read();
      clearTimeout(quiet);
      if (wentQuiet) throw new Error(`The stream sent nothing for ${(idleTimeoutMs ?? 0) / 1000} s.`);
      exhausted = done;
      buffer += decoder.decode(value, { stream: !done });
      let newline = buffer.indexOf("\n");
      while (newline !== -1) {
        const frame = assembler.line(buffer.slice(0, newline));
        buffer = buffer.slice(newline + 1);
        if (frame) yield frame;
        newline = buffer.indexOf("\n");
      }
    }
    // A last frame the server closed without a trailing blank line.
    if (buffer) {
      const frame = assembler.line(buffer);
      if (frame) yield frame;
    }
    const pending = assembler.flush();
    if (pending) yield pending;
  } finally {
    clearTimeout(quiet);
    if (!exhausted) await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
