import { isApiErrorBody, HttpError, toHttpError, normalizeError } from "../errors";

// ─── isApiErrorBody ───────────────────────────────────────────────────────────

describe("isApiErrorBody", () => {
  it("returns true for plain objects", () => {
    expect(isApiErrorBody({ detail: "err" })).toBe(true);
    expect(isApiErrorBody({})).toBe(true);
  });

  it("returns false for arrays", () => {
    expect(isApiErrorBody([])).toBe(false);
    expect(isApiErrorBody([{ detail: "err" }])).toBe(false);
  });

  it("returns false for null", () => {
    expect(isApiErrorBody(null)).toBe(false);
  });

  it("returns false for primitives", () => {
    expect(isApiErrorBody("string")).toBe(false);
    expect(isApiErrorBody(42)).toBe(false);
    expect(isApiErrorBody(true)).toBe(false);
    expect(isApiErrorBody(undefined)).toBe(false);
  });
});

// ─── HttpError ────────────────────────────────────────────────────────────────

describe("HttpError", () => {
  it("sets name, status, statusText, message, body", () => {
    const body = { detail: "Not found" };
    const err = new HttpError(404, "Not Found", "Not found", body);
    expect(err.name).toBe("HttpError");
    expect(err.status).toBe(404);
    expect(err.statusText).toBe("Not Found");
    expect(err.message).toBe("Not found");
    expect(err.body).toBe(body);
  });

  it("is instanceof Error", () => {
    expect(new HttpError(500, "ISE", "oops")).toBeInstanceOf(Error);
  });

  it("body is optional", () => {
    const err = new HttpError(204, "No Content", "empty");
    expect(err.body).toBeUndefined();
  });
});

// ─── toHttpError ─────────────────────────────────────────────────────────────

function makeResponse(status: number, body: unknown, statusText = "Error"): Response {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  return new Response(text, {
    status,
    statusText,
    headers: { "Content-Type": "application/json" },
  });
}

describe("toHttpError", () => {
  it("throws HttpError with detail from JSON body", async () => {
    const res = makeResponse(400, { detail: "Bad input" });
    await expect(toHttpError(res)).rejects.toMatchObject({
      status: 400,
      message: "Bad input",
    });
  });

  it("prefers detail over error over message", async () => {
    const res = makeResponse(400, { detail: "d", error: "e", message: "m" });
    await expect(toHttpError(res)).rejects.toMatchObject({ message: "d" });
  });

  it("falls back to error field when no detail", async () => {
    const res = makeResponse(400, { error: "e", message: "m" });
    await expect(toHttpError(res)).rejects.toMatchObject({ message: "e" });
  });

  it("falls back to message field", async () => {
    const res = makeResponse(400, { message: "m" });
    await expect(toHttpError(res)).rejects.toMatchObject({ message: "m" });
  });

  it("falls back to statusText when JSON has no known fields", async () => {
    const res = makeResponse(500, { foo: "bar" }, "Internal Server Error");
    await expect(toHttpError(res)).rejects.toMatchObject({ message: "Internal Server Error" });
  });

  it("handles plain text body", async () => {
    const res = new Response("plain error", { status: 503, statusText: "Service Unavailable" });
    await expect(toHttpError(res)).rejects.toMatchObject({ message: "plain error" });
  });

  it("falls back to statusText when body is empty", async () => {
    const res = new Response("", { status: 500, statusText: "Internal Server Error" });
    await expect(toHttpError(res)).rejects.toMatchObject({ message: "Internal Server Error" });
  });

  it("handles array body — uses JSON string as message (not statusText)", async () => {
    const arr = [{ detail: "ignored" }];
    const res = makeResponse(400, arr, "Bad Request");
    const err = await toHttpError(res).catch((e) => e);
    expect(err.message).toBe(JSON.stringify(arr));
    expect(err.body).toBe(JSON.stringify(arr));
  });

  it("attaches parsed body to HttpError", async () => {
    const res = makeResponse(422, { detail: "invalid" });
    const err = await toHttpError(res).catch((e) => e);
    expect(err.body).toEqual({ detail: "invalid" });
  });

  it("attaches string body when JSON parse fails", async () => {
    const res = new Response("not json", { status: 400 });
    const err = await toHttpError(res).catch((e) => e);
    expect(err.body).toBe("not json");
  });

  it("does not consume original response stream", async () => {
    const res = makeResponse(400, { detail: "err" });
    await toHttpError(res).catch(() => {});
    // original response body should still be readable
    await expect(res.text()).resolves.toBeDefined();
  });
});

// ─── normalizeError ───────────────────────────────────────────────────────────

describe("normalizeError", () => {
  it("re-throws HttpError unchanged", () => {
    const err = new HttpError(404, "Not Found", "not found");
    expect(() => normalizeError(err)).toThrow(err);
  });

  it("re-throws NEXT_REDIRECT error unchanged", () => {
    const err = new Error("NEXT_REDIRECT");
    expect(() => normalizeError(err)).toThrow(err);
  });

  it("maps TimeoutError to 'Request timed out.'", () => {
    const err = new DOMException("Timed out", "TimeoutError");
    expect(() => normalizeError(err)).toThrow("Request timed out.");
  });

  it("re-throws AbortError as-is (caller cancellation)", () => {
    const err = new DOMException("Aborted", "AbortError");
    expect(() => normalizeError(err)).toThrow(err);
  });

  it("maps TypeError to 'No connection to the server.'", () => {
    expect(() => normalizeError(new TypeError("Failed to fetch"))).toThrow(
      "No connection to the server."
    );
  });

  it("re-throws generic Error unchanged", () => {
    const err = new Error("something went wrong");
    expect(() => normalizeError(err)).toThrow(err);
  });

  it("wraps non-Error thrown string", () => {
    expect(() => normalizeError("raw string")).toThrow("raw string");
  });

  it("wraps non-Error thrown object", () => {
    expect(() => normalizeError({ code: 42 })).toThrow("[object Object]");
  });

  it("wraps null", () => {
    expect(() => normalizeError(null)).toThrow("null");
  });
});
