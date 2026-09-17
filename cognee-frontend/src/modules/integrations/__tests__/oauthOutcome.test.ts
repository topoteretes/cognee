import {
  describeOAuthFailure,
  isFailureOutcome,
  stashOAuthOutcome,
  takeOAuthOutcome,
} from "../oauthOutcome";

const STORAGE_KEY = "cognee-oauth-outcome";

describe("oauthOutcome", () => {
  beforeEach(() => {
    localStorage.clear();
    jest.restoreAllMocks();
  });

  describe("the popup-to-opener handoff", () => {
    it("carries the outcome across", () => {
      stashOAuthOutcome("slack", "error_already_connected");
      expect(takeOAuthOutcome("slack")).toBe("error_already_connected");
    });

    it("consumes the outcome, so a retry is not judged by the last attempt", () => {
      stashOAuthOutcome("slack", "error_exchange_failed");
      takeOAuthOutcome("slack");
      expect(takeOAuthOutcome("slack")).toBeNull();
    });

    it("returns null when the popup was closed by hand and wrote nothing", () => {
      expect(takeOAuthOutcome("slack")).toBeNull();
    });

    it("ignores an outcome belonging to a different provider", () => {
      stashOAuthOutcome("notion", "connected");
      expect(takeOAuthOutcome("slack")).toBeNull();
    });

    it("ignores a stale outcome no one consumed", () => {
      stashOAuthOutcome("slack", "connected");
      const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) as string);
      stored.at = Date.now() - 3 * 60 * 1000; // older than the 2-minute window
      localStorage.setItem(STORAGE_KEY, JSON.stringify(stored));

      expect(takeOAuthOutcome("slack")).toBeNull();
    });

    it("clears a stale outcome rather than leaving it to resurface", () => {
      stashOAuthOutcome("slack", "connected");
      takeOAuthOutcome("slack");
      expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    });

    it("writes nothing when the callback sent no outcome", () => {
      stashOAuthOutcome("slack", null);
      expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    });

    it("survives corrupted storage without throwing", () => {
      localStorage.setItem(STORAGE_KEY, "not json");
      expect(takeOAuthOutcome("slack")).toBeNull();
    });

    it("never lets a storage failure break the popup close", () => {
      jest.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
        throw new Error("quota exceeded");
      });
      expect(() => stashOAuthOutcome("slack", "connected")).not.toThrow();
    });

    it("survives storage being unreadable", () => {
      jest.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
        throw new Error("storage disabled");
      });
      expect(takeOAuthOutcome("slack")).toBeNull();
    });
  });

  describe("isFailureOutcome", () => {
    it("treats a successful install as no failure", () => {
      expect(isFailureOutcome("connected")).toBe(false);
    });

    it("treats a missing outcome as no failure — silence is correct there", () => {
      expect(isFailureOutcome(null)).toBe(false);
    });

    it.each(["cancelled", "error_invalid_state", "error_already_connected", "error_exchange_failed"])(
      "treats %s as a failure",
      (outcome) => {
        expect(isFailureOutcome(outcome)).toBe(true);
      },
    );

    it("treats an unrecognised outcome as a failure rather than assuming success", () => {
      expect(isFailureOutcome("error_something_new")).toBe(true);
    });
  });

  describe("describeOAuthFailure", () => {
    it("says the workspace is taken, and that it must be freed elsewhere", () => {
      const message = describeOAuthFailure("error_already_connected", "Slack");
      expect(message).toContain("already connected");
      expect(message).toContain("disconnect it there first");
    });

    it("does not invite a retry for a conflict no retry can fix", () => {
      expect(describeOAuthFailure("error_already_connected", "Slack")).not.toMatch(/try again/i);
    });

    it("says the link expired, and to start again", () => {
      const message = describeOAuthFailure("error_invalid_state", "Slack");
      expect(message).toContain("expired");
    });

    it("invites a retry when retrying can plausibly work", () => {
      expect(describeOAuthFailure("error_exchange_failed", "Slack")).toMatch(/try once more/i);
    });

    it("does not call a deliberate cancel an error", () => {
      const message = describeOAuthFailure("cancelled", "Slack");
      expect(message).toContain("cancelled");
      expect(message).not.toMatch(/failed|error/i);
    });

    it("names the provider it is talking about", () => {
      expect(describeOAuthFailure("error_exchange_failed", "Notion")).toContain("Notion");
    });

    it("stays reportable for an outcome the backend adds later", () => {
      const message = describeOAuthFailure("error_brand_new", "Slack");
      expect(message).toContain("error_brand_new");
    });
  });
});
