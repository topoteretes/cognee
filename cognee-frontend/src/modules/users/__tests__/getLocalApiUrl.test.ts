/**
 * The window.location fallback is the point of this suite, so pin jsdom to a
 * non-localhost origin: replaceState cannot cross origins from the default one.
 *
 * @jest-environment-options {"url": "http://127.0.0.1:3000/local-login"}
 */
import { getLocalApiUrl } from "../getLocalApiUrl";
import { RUNTIME_CONFIG_ELEMENT_ID } from "@/modules/config/runtimeConfig";

function renderRuntimeConfig(backendUrl: string | null): void {
  const element = document.createElement("script");
  element.id = RUNTIME_CONFIG_ELEMENT_ID;
  element.type = "application/json";
  element.textContent = JSON.stringify({ backendUrl });
  document.head.appendChild(element);
}

describe("getLocalApiUrl", () => {
  const originalUrl = window.location.href;
  const originalOverride = process.env.NEXT_PUBLIC_LOCAL_API_URL;

  afterEach(() => {
    window.history.replaceState({}, "", originalUrl);
    document.getElementById(RUNTIME_CONFIG_ELEMENT_ID)?.remove();
    if (originalOverride === undefined) {
      delete process.env.NEXT_PUBLIC_LOCAL_API_URL;
    } else {
      process.env.NEXT_PUBLIC_LOCAL_API_URL = originalOverride;
    }
  });

  it("uses the browser host for the local API by default", () => {
    window.history.replaceState({}, "", "http://127.0.0.1:3000/local-login");

    expect(getLocalApiUrl()).toBe("http://127.0.0.1:8000");
  });

  it("keeps an explicitly configured API URL unchanged", () => {
    process.env.NEXT_PUBLIC_LOCAL_API_URL = "https://api.example.com";

    expect(getLocalApiUrl()).toBe("https://api.example.com");
  });

  it("prefers the server-injected runtime config over the build-time value", () => {
    process.env.NEXT_PUBLIC_LOCAL_API_URL = "https://baked-in.example.com";
    renderRuntimeConfig("https://from-docker-run.example.com");

    expect(getLocalApiUrl()).toBe("https://from-docker-run.example.com");
  });

  it("falls through to the build-time value when the image ships no runtime config", () => {
    process.env.NEXT_PUBLIC_LOCAL_API_URL = "https://baked-in.example.com";
    renderRuntimeConfig(null);

    expect(getLocalApiUrl()).toBe("https://baked-in.example.com");
  });

  it("normalises trailing slashes from either source", () => {
    renderRuntimeConfig("https://api.example.com/");
    expect(getLocalApiUrl()).toBe("https://api.example.com");

    document.getElementById(RUNTIME_CONFIG_ELEMENT_ID)?.remove();
    process.env.NEXT_PUBLIC_LOCAL_API_URL = "https://api.example.com/";
    expect(getLocalApiUrl()).toBe("https://api.example.com");
  });
});
