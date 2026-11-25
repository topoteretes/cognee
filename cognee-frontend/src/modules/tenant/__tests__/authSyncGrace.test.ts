import { AUTH_SYNC_GRACE_MS, authErrorRunStart, isAuthError, isWithinAuthSyncGrace } from "../authSyncGrace";
import { HttpError } from "@/services/http";

const unauthorized = new HttpError(401, "Unauthorized", "key not synced yet");

describe("authSyncGrace", () => {
  describe("isAuthError", () => {
    it.each([401, 403])("treats a %i as an auth error", (status) => {
      expect(isAuthError(new HttpError(status, "Unauthorized", "rejected"))).toBe(true);
    });

    it.each([
      ["a 500", new HttpError(500, "Internal Server Error", "boom")],
      ["a network error", new Error("ECONNREFUSED")],
      ["a non-error value", "401"],
    ])("does not treat %s as an auth error", (_label, err) => {
      expect(isAuthError(err)).toBe(false);
    });
  });

  describe("authErrorRunStart", () => {
    it("starts a run at the first auth error", () => {
      expect(authErrorRunStart(null, unauthorized, 1_000)).toBe(1_000);
    });

    it("keeps the run's start across further auth errors", () => {
      expect(authErrorRunStart(1_000, unauthorized, 9_000)).toBe(1_000);
    });

    it("ends the run on any other failure", () => {
      expect(authErrorRunStart(1_000, new Error("ECONNREFUSED"), 9_000)).toBeNull();
    });
  });

  describe("isWithinAuthSyncGrace", () => {
    it("stays closed while neither clock has started", () => {
      expect(isWithinAuthSyncGrace({ authErrorRunStartedAt: null, provisioningSeenAt: null }, 1_000)).toBe(false);
    });

    it("closes AUTH_SYNC_GRACE_MS after the run of auth errors began", () => {
      const clocks = { authErrorRunStartedAt: 0, provisioningSeenAt: null };
      expect(isWithinAuthSyncGrace(clocks, AUTH_SYNC_GRACE_MS - 1)).toBe(true);
      expect(isWithinAuthSyncGrace(clocks, AUTH_SYNC_GRACE_MS)).toBe(false);
    });

    it("closes AUTH_SYNC_GRACE_MS after the last provisioning sighting", () => {
      const clocks = { authErrorRunStartedAt: null, provisioningSeenAt: 0 };
      expect(isWithinAuthSyncGrace(clocks, AUTH_SYNC_GRACE_MS - 1)).toBe(true);
      expect(isWithinAuthSyncGrace(clocks, AUTH_SYNC_GRACE_MS)).toBe(false);
    });

    it("stays open while either clock is", () => {
      expect(isWithinAuthSyncGrace({ authErrorRunStartedAt: 0, provisioningSeenAt: 50_000 }, AUTH_SYNC_GRACE_MS + 10_000)).toBe(true);
    });
  });
});
