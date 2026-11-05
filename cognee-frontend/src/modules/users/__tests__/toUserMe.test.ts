import toUserMe from "../toUserMe";
import type { UserMe } from "../UserContext";

const FULL: UserMe = {
  isSeenWelcome: true,
  onboardingCompletedAt: "2026-01-01T00:00:00Z",
  activeWorkspaceId: "ws-1",
  pendingFeatureAnnouncements: ["a"],
  userId: "ignored",
  accountCreatedAt: "2025-01-01T00:00:00Z",
  sessionRecordingOptIn: true,
  sessionRecordingConsentAnswered: true,
  profile: { firstName: "Ada", lastName: "Lovelace" },
};

// The backend half of session-recording consent ships separately, so a
// response without either field is a real deployment state, not a malformed
// one.
function withoutConsentFields(): UserMe {
  const { sessionRecordingOptIn, sessionRecordingConsentAnswered, ...rest } = FULL;
  void sessionRecordingOptIn;
  void sessionRecordingConsentAnswered;
  return rest as UserMe;
}

describe("toUserMe", () => {
  it("takes userId from the session rather than the backend payload", () => {
    expect(toUserMe(FULL, "auth0|42").userId).toBe("auth0|42");
    expect(toUserMe(FULL, null).userId).toBeNull();
  });

  it("carries every field through unchanged when the backend sends them all", () => {
    expect(toUserMe(FULL, "auth0|42")).toEqual({ ...FULL, userId: "auth0|42" });
  });

  it("reads an absent opt-in as no consent", () => {
    // Absent and false are the same decision here — do not record — so the
    // safe collapse is the correct one.
    expect(toUserMe(withoutConsentFields(), null).sessionRecordingOptIn).toBe(false);
  });

  it("reads an absent answered flag as unknown, not as unanswered", () => {
    // The opposite collapse to the one above, because these are opposite
    // instructions: false means ask. Defaulting to false would ask every user
    // a question the deployment cannot store an answer to, behind a modal
    // that has no dismissal.
    expect(toUserMe(withoutConsentFields(), null).sessionRecordingConsentAnswered).toBeNull();
  });

  it("keeps an explicit false as a request to ask", () => {
    expect(
      toUserMe({ ...FULL, sessionRecordingConsentAnswered: false }, null)
        .sessionRecordingConsentAnswered,
    ).toBe(false);
  });

  describe("profile (CLO-456)", () => {
    it("carries the stored names through", () => {
      expect(toUserMe(FULL, null).profile).toEqual({ firstName: "Ada", lastName: "Lovelace" });
    });

    it("reads a backend that predates the field as unknown, not as a user with no name", () => {
      // null makes the profile form read-only. A profile of nulls would leave it
      // editable, and a save would go to an endpoint that is not deployed.
      const { profile, ...rest } = FULL;
      void profile;

      expect(toUserMe(rest as UserMe, null).profile).toBeNull();
    });

    it("keeps an account with no name set as a profile of nulls, which stays editable", () => {
      expect(
        toUserMe({ ...FULL, profile: { firstName: null, lastName: null } }, null).profile,
      ).toEqual({ firstName: null, lastName: null });
    });

    it("reads a name that is not a string as not set rather than passing it on", () => {
      const odd = { ...FULL, profile: { firstName: 7, lastName: {} } } as unknown as UserMe;

      expect(toUserMe(odd, null).profile).toEqual({ firstName: null, lastName: null });
    });

    it.each([null, "Ada", 7, []])("reads a profile that is %p as unknown", (value) => {
      const odd = { ...FULL, profile: value } as unknown as UserMe;

      expect(toUserMe(odd, null).profile).toBeNull();
    });
  });
});
