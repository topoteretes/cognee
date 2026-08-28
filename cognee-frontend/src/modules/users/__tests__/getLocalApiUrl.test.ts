/**
 * The window.location fallback is the point of this suite, so pin jsdom to a
 * non-localhost origin: replaceState cannot cross origins from the default one.
 *
 * @jest-environment-options {"url": "http://127.0.0.1:3000/local-login"}
 */
import { getLocalApiUrl } from "../getLocalApiUrl";

describe("getLocalApiUrl", () => {
  const originalUrl = window.location.href;
  const originalOverride = process.env.NEXT_PUBLIC_LOCAL_API_URL;

  afterEach(() => {
    window.history.replaceState({}, "", originalUrl);
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
});
