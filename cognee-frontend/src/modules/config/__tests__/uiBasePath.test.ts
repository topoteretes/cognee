import { uiPath } from "../uiBasePath";

const original = process.env.NEXT_PUBLIC_UI_BASE_PATH;
afterEach(() => {
  if (original === undefined) delete process.env.NEXT_PUBLIC_UI_BASE_PATH;
  else process.env.NEXT_PUBLIC_UI_BASE_PATH = original;
});

it.each([undefined, "", "/cognee", "/tools/cognee"])("prefixes raw URLs with %s", (base) => {
  if (base === undefined) delete process.env.NEXT_PUBLIC_UI_BASE_PATH;
  else process.env.NEXT_PUBLIC_UI_BASE_PATH = base;
  for (const path of ["/", "/local-login", "/api/local-signout", "/images/icons/avatar.svg", "/videos/full-demo.mp4", "/api/visualize?dataset_id=demo"]) {
    expect(uiPath(path)).toBe((base ?? "") + path);
  }
});
it.each(["/", "/cognee/", "//host", "https://host", "/../admin", "/a?b", "/a#b"])("rejects invalid base %s", (base) => {
  process.env.NEXT_PUBLIC_UI_BASE_PATH = base;
  expect(() => uiPath("/local-login")).toThrow("Invalid local UI path");
});
it.each(["", "relative", "//host/path", "https://host/path"])("rejects non-local URL %s", (path) => {
  process.env.NEXT_PUBLIC_UI_BASE_PATH = "/cognee";
  expect(() => uiPath(path)).toThrow("Invalid local UI path");
});

it.each(["", "/cognee", "/tools/cognee"])("keeps the opt-in same-origin API proxy under %s", async (base) => {
  const originalProxy = process.env.NEXT_PUBLIC_API_SAME_ORIGIN;
  try {
    process.env.NEXT_PUBLIC_API_SAME_ORIGIN = "true";
    process.env.NEXT_PUBLIC_UI_BASE_PATH = base;
    const { getLocalApiUrl } = await import("@/modules/users/getLocalApiUrl");
    expect(getLocalApiUrl()).toBe(window.location.origin + base);
  } finally {
    if (originalProxy === undefined) delete process.env.NEXT_PUBLIC_API_SAME_ORIGIN;
    else process.env.NEXT_PUBLIC_API_SAME_ORIGIN = originalProxy;
  }
});
