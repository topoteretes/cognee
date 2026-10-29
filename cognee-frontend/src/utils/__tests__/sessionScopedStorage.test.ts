import {
  SESSION_SCOPED_COOKIES,
  SESSION_SCOPED_STORAGE_KEYS,
  SESSION_SCOPED_STORAGE_PREFIXES,
  SIGNUP_TRACKED_FLAG,
  FIRST_LOGIN_CHECK_FLAG,
  CONSENT_REPORTED_FLAG,
  clearSessionScopedStorage,
  ensureSessionFingerprint,
} from "../sessionScopedStorage";
import { getSearchType, setSearchType } from "../browserStorage";

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  document.cookie = "cognee_selected_tenant=abc;path=/";
});

describe("clearSessionScopedStorage", () => {
  it("removes every known session-scoped key from both storages", () => {
    for (const key of SESSION_SCOPED_STORAGE_KEYS) {
      localStorage.setItem(key, "stale");
      sessionStorage.setItem(key, "stale");
    }

    clearSessionScopedStorage();

    for (const key of SESSION_SCOPED_STORAGE_KEYS) {
      expect(localStorage.getItem(key)).toBeNull();
      expect(sessionStorage.getItem(key)).toBeNull();
    }
  });

  // Iterates the registry rather than naming one prefix, so a prefix added
  // later is covered the moment it is registered.
  it.each(SESSION_SCOPED_STORAGE_PREFIXES)("removes per-tenant keys matched by prefix %s", (prefix) => {
    localStorage.setItem(`${prefix}tenant-123`, "stale");
    sessionStorage.setItem(`${prefix}tenant-123`, "stale");

    clearSessionScopedStorage();

    expect(localStorage.getItem(`${prefix}tenant-123`)).toBeNull();
    expect(sessionStorage.getItem(`${prefix}tenant-123`)).toBeNull();
  });

  // Written and read through the accessor the Search page uses, not the
  // registry: the registry test above passes for any prefix that is listed,
  // and says nothing about whether the key actually in use is one of them.
  it("clears the remembered Search type, which picks the retriever for the next account", () => {
    setSearchType("tenant-123", "code");

    clearSessionScopedStorage();

    expect(getSearchType("tenant-123")).toBeNull();
  });

  it("does not touch keys outside the session-scoped list", () => {
    localStorage.setItem("cognee-pipeline-settings", "keep-me");
    sessionStorage.setItem("cognee_anonymous_id", "keep-me");

    clearSessionScopedStorage();

    expect(localStorage.getItem("cognee-pipeline-settings")).toBe("keep-me");
    expect(sessionStorage.getItem("cognee_anonymous_id")).toBe("keep-me");
  });

  it("expires every account-scoped cookie on the domain it was written with", () => {
    // jsdom cannot store a .cognee.ai cookie, so the emitted strings are
    // captured: the domain is the load-bearing part, since a delete that omits
    // it mints a host-only sibling and leaves the real cookie in place.
    const writes: string[] = [];
    const spy = jest.spyOn(document, "cookie", "set").mockImplementation((value: string) => {
      writes.push(value);
    });

    clearSessionScopedStorage();
    spy.mockRestore();

    for (const { name, domain } of SESSION_SCOPED_COOKIES) {
      const expected = `${name}=;Max-Age=0;path=/${domain ? `;domain=${domain}` : ""}`;
      expect(writes).toContain(expected);
    }
  });

  it("leaves the sign-up cookies alone, because a switch cannot tell whose they are", () => {
    // An account switch looks identical whether the pending consent answer
    // belongs to a previous visitor or to the account now arriving, and
    // clearing it here destroyed the second kind. Whose answer it is decided
    // where it is written instead: the email path binds it to the address that
    // gave it and the flush refuses to write on a mismatch. Sign-out still
    // clears them (see the signout route's own test).
    const writes: string[] = [];
    const spy = jest.spyOn(document, "cookie", "set").mockImplementation((value: string) => {
      writes.push(value);
    });

    clearSessionScopedStorage();
    spy.mockRestore();

    const cleared = writes.join("\n");
    expect(cleared).not.toContain("cognee_marketing_consent");
    expect(cleared).not.toContain("cognee_signup_intent");
  });

  // All three store the user id they belong to, so a second account is a
  // different key and nothing needs clearing. Clearing was actively harmful:
  // UserProvider is an ancestor of the component that sets them and React
  // commits child effects first, so the wipe landed immediately after the write
  // and the next mount emitted a second `Signed Up`. Asserted per flag rather
  // than for the one that happened to be named: only SIGNUP_TRACKED_FLAG used
  // to be pinned, and putting either of the others back into the list left the
  // whole suite green.
  it.each([
    ["SIGNUP_TRACKED_FLAG", SIGNUP_TRACKED_FLAG],
    ["FIRST_LOGIN_CHECK_FLAG", FIRST_LOGIN_CHECK_FLAG],
    ["CONSENT_REPORTED_FLAG", CONSENT_REPORTED_FLAG],
  ])("leaves %s alone, because clearing it duplicated the event", (_name, flag) => {
    expect(SESSION_SCOPED_STORAGE_KEYS).not.toContain(flag);

    // localStorage is where they actually live: a signup happens once per
    // account, not once per tab, and a second tab used to emit a duplicate.
    localStorage.setItem(flag, "auth0|someone");
    sessionStorage.setItem(flag, "auth0|someone");
    clearSessionScopedStorage();

    expect(localStorage.getItem(flag)).toBe("auth0|someone");
    expect(sessionStorage.getItem(flag)).toBe("auth0|someone");
  });

  it("clears the tenant cookie", () => {
    clearSessionScopedStorage();

    expect(document.cookie).not.toContain("cognee_selected_tenant=abc");
  });
});

describe("ensureSessionFingerprint", () => {
  it("wipes session-scoped storage when a different account is detected", () => {
    localStorage.setItem("cognee_init_cache", "user-a-cache");
    ensureSessionFingerprint("auth0|user-a");

    ensureSessionFingerprint("auth0|user-b");

    expect(localStorage.getItem("cognee_init_cache")).toBeNull();
  });

  it("does not wipe storage for the same account across calls", () => {
    ensureSessionFingerprint("auth0|user-a");
    localStorage.setItem("cognee_init_cache", "user-a-cache");

    ensureSessionFingerprint("auth0|user-a");

    expect(localStorage.getItem("cognee_init_cache")).toBe("user-a-cache");
  });

  it("does nothing when userId is null or undefined (identity not resolved yet)", () => {
    localStorage.setItem("cognee_init_cache", "keep-me");

    ensureSessionFingerprint(null);
    ensureSessionFingerprint(undefined);

    expect(localStorage.getItem("cognee_init_cache")).toBe("keep-me");
  });
});
