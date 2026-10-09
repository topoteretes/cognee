import { isAuthRoutePath, isSessionlessPath } from "../sessionRoutes";

describe("isAuthRoutePath", () => {
  it("matches an auth route exactly", () => {
    expect(isAuthRoutePath("/sign-in")).toBe(true);
    expect(isAuthRoutePath("/verify-email")).toBe(true);
  });

  it("matches auth sub-routes on a path boundary", () => {
    expect(isAuthRoutePath("/sign-in/callback")).toBe(true);
  });

  it("does NOT match a route merely prefixed by an auth path", () => {
    expect(isAuthRoutePath("/sign-in-legacy")).toBe(false);
    expect(isAuthRoutePath("/sign-upgrade")).toBe(false);
  });

  it("does not match protected routes", () => {
    expect(isAuthRoutePath("/dashboard")).toBe(false);
    expect(isAuthRoutePath("/")).toBe(false);
  });
});

describe("isSessionlessPath", () => {
  it("includes auth routes", () => {
    expect(isSessionlessPath("/sign-in")).toBe(true);
  });

  it("includes public no-session pages", () => {
    expect(isSessionlessPath("/waitlist")).toBe(true);
  });

  it("does not include protected routes", () => {
    expect(isSessionlessPath("/dashboard")).toBe(false);
    expect(isSessionlessPath("/welcome")).toBe(false);
  });
});
