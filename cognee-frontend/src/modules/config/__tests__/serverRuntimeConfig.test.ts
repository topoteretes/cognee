import { collectRuntimeConfig, getServerBackendUrl } from "../serverRuntimeConfig";

describe("server backend URL resolution", () => {
  const original = {
    runtime: process.env.COGNEE_BACKEND_URL,
    buildTime: process.env.NEXT_PUBLIC_LOCAL_API_URL,
  };

  afterEach(() => {
    for (const [key, value] of [
      ["COGNEE_BACKEND_URL", original.runtime],
      ["NEXT_PUBLIC_LOCAL_API_URL", original.buildTime],
    ] as const) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  });

  it("defaults to the local backend when nothing is configured", () => {
    delete process.env.COGNEE_BACKEND_URL;
    delete process.env.NEXT_PUBLIC_LOCAL_API_URL;

    expect(getServerBackendUrl()).toBe("http://localhost:8000");
  });

  it("still honours a URL that was baked in at build time", () => {
    delete process.env.COGNEE_BACKEND_URL;
    process.env.NEXT_PUBLIC_LOCAL_API_URL = "http://baked-in:8000";

    expect(getServerBackendUrl()).toBe("http://baked-in:8000");
  });

  it("prefers the runtime variable, which is the one a container can set", () => {
    process.env.COGNEE_BACKEND_URL = "http://from-docker-run:9000/";
    process.env.NEXT_PUBLIC_LOCAL_API_URL = "http://baked-in:8000";

    expect(getServerBackendUrl()).toBe("http://from-docker-run:9000");
  });

  it("fails loudly instead of silently falling back", () => {
    process.env.COGNEE_BACKEND_URL = "not-a-url";

    expect(() => getServerBackendUrl()).toThrow(/COGNEE_BACKEND_URL/);
  });
});

describe("collectRuntimeConfig", () => {
  const original = {
    runtime: process.env.COGNEE_BACKEND_URL,
    buildTime: process.env.NEXT_PUBLIC_LOCAL_API_URL,
  };

  afterEach(() => {
    for (const [key, value] of [
      ["COGNEE_BACKEND_URL", original.runtime],
      ["NEXT_PUBLIC_LOCAL_API_URL", original.buildTime],
    ] as const) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  });

  it("publishes nothing when no runtime URL is set, leaving the browser its own fallback", () => {
    delete process.env.COGNEE_BACKEND_URL;
    process.env.NEXT_PUBLIC_LOCAL_API_URL = "http://baked-in:8000";

    expect(collectRuntimeConfig()).toEqual({ backendUrl: null });
  });

  it("publishes the runtime URL when one is set", () => {
    process.env.COGNEE_BACKEND_URL = "https://cognee.example.com";

    expect(collectRuntimeConfig()).toEqual({ backendUrl: "https://cognee.example.com" });
  });
});
