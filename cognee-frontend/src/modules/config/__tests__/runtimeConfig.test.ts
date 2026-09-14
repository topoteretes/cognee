import {
  RUNTIME_CONFIG_ELEMENT_ID,
  normalizeBackendUrl,
  serializeRuntimeConfig,
  stripTrailingSlash,
  readRuntimeConfig,
} from "../runtimeConfig";

function renderConfigElement(json: string): void {
  const element = document.createElement("script");
  element.id = RUNTIME_CONFIG_ELEMENT_ID;
  element.type = "application/json";
  element.textContent = json;
  document.head.appendChild(element);
}

describe("normalizeBackendUrl", () => {
  it("treats an unset, empty or blank value as not configured", () => {
    expect(normalizeBackendUrl(undefined, "TEST_ENV")).toBeNull();
    expect(normalizeBackendUrl("", "TEST_ENV")).toBeNull();
    expect(normalizeBackendUrl("   ", "TEST_ENV")).toBeNull();
  });

  it("strips trailing slashes so concatenated paths stay single-slashed", () => {
    expect(normalizeBackendUrl("http://cognee:8000/", "TEST_ENV")).toBe("http://cognee:8000");
    expect(normalizeBackendUrl("http://cognee:8000///", "TEST_ENV")).toBe("http://cognee:8000");
  });

  it("rejects a host:port written without a scheme, the likeliest mistake", () => {
    expect(() => normalizeBackendUrl("cognee:8000", "COGNEE_BACKEND_URL")).toThrow(
      /COGNEE_BACKEND_URL must be an absolute http\(s\) URL, got "cognee:8000"/,
    );
  });

  it("rejects protocols the browser cannot fetch from", () => {
    expect(() => normalizeBackendUrl("ftp://cognee:8000", "COGNEE_BACKEND_URL")).toThrow(
      /must be an absolute http\(s\) URL/,
    );
  });

  it("rejects a value that only looks like a URL", () => {
    expect(() => normalizeBackendUrl("http://", "COGNEE_BACKEND_URL")).toThrow(
      /COGNEE_BACKEND_URL is not a valid URL/,
    );
  });
});

describe("serializeRuntimeConfig", () => {
  it("escapes sequences that would break out of an inline script", () => {
    const serialized = serializeRuntimeConfig({
      backendUrl: "http://x/</script><script>alert(1)</script>",
    });

    expect(serialized).not.toContain("</script>");
    expect(serialized).toContain("\\u003c");
    expect(serialized).toContain("\\u003e");
  });

  it("escapes the line terminators that are legal in JSON but not in JS source", () => {
    const serialized = serializeRuntimeConfig({
      backendUrl: "http://x/\u2028\u2029",
    });

    expect(serialized).not.toContain("\u2028");
    expect(serialized).not.toContain("\u2029");
    expect(serialized).toContain("\\u2028");
    expect(serialized).toContain("\\u2029");
  });

  it("round-trips through the same eval the browser performs", () => {
    const config = { backendUrl: "https://cognee.example.com" };

    expect(JSON.parse(serializeRuntimeConfig(config))).toEqual(config);
  });
});

describe("readRuntimeConfig", () => {
  afterEach(() => {
    document.getElementById(RUNTIME_CONFIG_ELEMENT_ID)?.remove();
  });

  it("is empty when the server rendered nothing", () => {
    expect(readRuntimeConfig()).toEqual({});
  });

  it("returns what the server rendered", () => {
    renderConfigElement(JSON.stringify({ backendUrl: "http://cognee:8000" }));

    expect(readRuntimeConfig()).toEqual({ backendUrl: "http://cognee:8000" });
  });

  it("degrades to the caller's fallback rather than throwing on a corrupt payload", () => {
    renderConfigElement("{not json");

    expect(readRuntimeConfig()).toEqual({});
  });

  it("rejects a scheme the browser would treat as executable", () => {
    // This value reaches href attributes, so an unvalidated javascript: URL
    // taken straight out of the DOM would be an XSS sink.
    renderConfigElement(JSON.stringify({ backendUrl: "javascript:alert(1)" }));

    expect(readRuntimeConfig()).toEqual({});
  });

  it("rejects a scheme the client cannot fetch from", () => {
    renderConfigElement(JSON.stringify({ backendUrl: "ftp://cognee:8000" }));

    expect(readRuntimeConfig()).toEqual({});
  });

  it("normalises a trailing slash coming out of the DOM", () => {
    renderConfigElement(JSON.stringify({ backendUrl: "http://cognee:8000/" }));

    expect(readRuntimeConfig()).toEqual({ backendUrl: "http://cognee:8000" });
  });

  it("rebuilds the URL rather than passing the document's string through", () => {
    // The scheme comes from a literal, never from the document, which is what
    // makes the value safe to put in an href.
    renderConfigElement(JSON.stringify({ backendUrl: "HTTPS://Cognee.Example.com:8443/api" }));

    expect(readRuntimeConfig()).toEqual({ backendUrl: "https://cognee.example.com:8443/api" });
  });

  it("drops query and fragment, which a backend base URL has no use for", () => {
    renderConfigElement(JSON.stringify({ backendUrl: "http://cognee:8000/?a=1#x" }));

    expect(readRuntimeConfig()).toEqual({ backendUrl: "http://cognee:8000" });
  });
});

describe("stripTrailingSlash", () => {
  it("leaves a bare origin alone", () => {
    expect(stripTrailingSlash("http://cognee:8000")).toBe("http://cognee:8000");
  });
});
