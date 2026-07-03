import { describe, expect, it } from "vitest";

import { normalizeEndpoint, resolveSearchType, validateConfig, type CogneeConfig } from "./config";

const BASE: CogneeConfig = {
  endpoint: "http://localhost:8011",
  searchType: "auto",
  topK: 15,
  includeReferences: true,
  respectGitignore: true,
  maxFileSizeKb: 512,
  requestTimeoutMs: 300000,
};

describe("validateConfig", () => {
  it("accepts a valid configuration", () => {
    expect(validateConfig(BASE)).toEqual({ ok: true, errors: [] });
  });

  it("rejects an endpoint without an http(s) scheme", () => {
    expect(validateConfig({ ...BASE, endpoint: "localhost:8011" }).ok).toBe(false);
  });

  it("rejects an out-of-range topK", () => {
    expect(validateConfig({ ...BASE, topK: 0 }).ok).toBe(false);
    expect(validateConfig({ ...BASE, topK: 500 }).ok).toBe(false);
  });

  it("rejects non-positive timeouts and sizes", () => {
    expect(validateConfig({ ...BASE, requestTimeoutMs: 0 }).ok).toBe(false);
    expect(validateConfig({ ...BASE, maxFileSizeKb: -1 }).ok).toBe(false);
  });
});

describe("resolveSearchType", () => {
  it("maps auto (or empty) to null for server-side routing", () => {
    expect(resolveSearchType("auto")).toBeNull();
    expect(resolveSearchType("")).toBeNull();
  });

  it("passes an explicit strategy through", () => {
    expect(resolveSearchType("CHUNKS")).toBe("CHUNKS");
  });
});

describe("normalizeEndpoint", () => {
  it("trims whitespace and trailing slashes", () => {
    expect(normalizeEndpoint("  http://localhost:8011///  ")).toBe("http://localhost:8011");
  });
});
