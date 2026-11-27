/** @jest-environment node */
import type { NextRequest } from "next/server";
import { GET as visualize } from "../visualize/route";
import { GET as inventory } from "../schema/inventory/route";
import { GET as provenance } from "../schema/provenance/route";
import { GET as legacyProvenance } from "../schema-provenance/route";
import { GET as signout } from "../local-signout/route";

// Response serialization is Next's responsibility; exercise the handlers' real
// backend requests and URL resolver against fetchMock.
jest.mock("next/server", () => ({
  NextResponse: class extends Response {
    cookies = { set: jest.fn() };
    static json(value: unknown, init?: ResponseInit) {
      return new Response(JSON.stringify(value), init);
    }
    static redirect(url: URL) {
      return new this(null, { status: 307, headers: { location: url.toString() } });
    }
  },
}));

const routes = [visualize, inventory, provenance, legacyProvenance];
const originalEnv = { ...process.env };
function request(headers: Record<string, string> = {}): NextRequest {
  return Object.assign(new Request("http://ui.test/api/test?dataset_id=demo", { headers }), {
    nextUrl: new URL("http://ui.test/api/test?dataset_id=demo"),
  }) as NextRequest;
}

beforeEach(() => {
  fetchMock.resetMocks();
  process.env.COGNEE_BACKEND_URL = "http://configured.test:9100/";
  jest.spyOn(console, "log").mockImplementation(() => {});
  jest.spyOn(console, "error").mockImplementation(() => {});
});
afterEach(() => {
  process.env = { ...originalEnv };
  jest.restoreAllMocks();
});

it.each(routes)("sends an anonymous request on as-is, without logging in (%#)", async (route) => {
  fetchMock.mockResponseOnce("", { status: 401 });
  const response = await route(request());
  expect(response.status).toBe(401);
  expect(fetchMock).toHaveBeenCalledTimes(1);
  expect(fetchMock.mock.calls[0][0]).toMatch(/^http:\/\/configured.test:9100\/api\/v1\//);
  expect(fetchMock.mock.calls[0][1]?.headers).toEqual({});
});

it.each(routes.flatMap(route => ["cookie", "authorization", "x-api-key"].map(header => ({ route, header }))))(
  "forwards $header", async ({ route, header }) => {
    fetchMock.mockResponseOnce("{}");
    await route(request({ [header]: "user-credential" }));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][1]?.headers).toEqual({ [header]: "user-credential" });
    expect(fetchMock.mock.calls[0][0]).toMatch(/^http:\/\/configured.test:9100\/api\/v1\//);
  },
);

it("logs out against the runtime backend and passes its cookie deletions on", async () => {
  // The backend names the cookies; the browser only drops the ones it is told to.
  const deletions = ['auth_token=""; Max-Age=0; Path=/', 'auth_token_refresh=""; Max-Age=0; Path=/'];
  // Headers.getSetCookie is in Node's fetch, which the route runs on, but not in jest-fetch-mock's.
  fetchMock.mockImplementationOnce(
    async () => ({ headers: { getSetCookie: () => deletions } }) as unknown as Response,
  );
  const response = await signout(request({ cookie: "session=test" }));
  expect(fetchMock).toHaveBeenCalledWith("http://configured.test:9100/api/v1/auth/logout", {
    method: "POST", headers: { cookie: "session=test" },
  });
  expect(response.status).toBe(307);
  expect(response.headers.get("location")).toBe("http://ui.test/local-login");
  expect(response.headers.get("set-cookie")).toBe(deletions.join(", "));
});

it("still sends the browser to the login page when the backend is down", async () => {
  fetchMock.mockRejectOnce(new TypeError("fetch failed"));
  const response = await signout(request({ cookie: "session=test" }));
  expect(response.status).toBe(307);
  expect(response.headers.get("location")).toBe("http://ui.test/local-login");
  expect(response.headers.get("set-cookie")).toBeNull();
});
