import type { UserMe } from "@/modules/users/UserContext";

/**
 * The one projection of the backend's /users/me onto the client's UserMe.
 *
 * Every route that returns this shape (GET /api/me and each PATCH, whose
 * responses UserProvider writes straight into the ME query cache) goes through
 * here, and each used to spell the field list out for itself. The one reader that
 * does not is getServerUserMe, which only needs the two facts the redirects
 * decide on. A field added to UserMe but
 * forgotten in one of them does not fail loudly: that route's response simply
 * drops it, so a PATCH silently rewrites the cache with the field missing.
 * For session-recording consent that reads as "no consent" and would stop the
 * replay tools mid-session; for the next field it will read as something else.
 *
 * `userId` comes from the Auth0 session rather than the backend, which is why
 * it stays a parameter.
 */
export default function toUserMe(data: UserMe, userId: string | null): UserMe {
  // The backend nests the names under `profile`. Absent (a backend that predates
  // the field) is null and not a profile of nulls: null makes the profile form
  // read-only, while a profile of nulls is an account with no name yet, whose form
  // is editable. Collapsing one into the other would offer a save to an endpoint
  // that is not there.
  const names = data.profile;
  const profile =
    names && typeof names === "object" && !Array.isArray(names)
      ? {
          firstName: typeof names.firstName === "string" ? names.firstName : null,
          lastName: typeof names.lastName === "string" ? names.lastName : null,
        }
      : null;

  return {
    isSeenWelcome: data.isSeenWelcome,
    onboardingCompletedAt: data.onboardingCompletedAt,
    activeWorkspaceId: data.activeWorkspaceId,
    pendingFeatureAnnouncements: data.pendingFeatureAnnouncements,
    userId,
    accountCreatedAt: data.accountCreatedAt,
    // Never defaulted to true on a partial response: this value decides
    // whether session replay runs, so an absent field has to read as no
    // consent rather than as consent.
    sessionRecordingOptIn: data.sessionRecordingOptIn === true,
    // Absent is null, not false. For optIn above, absent and false are the
    // same decision — do not record — so collapsing them is safe. Here they
    // are opposite instructions: false means ask, and a backend that has
    // never heard of this field would be asking every user a question it
    // cannot store the answer to.
    sessionRecordingConsentAnswered:
      typeof data.sessionRecordingConsentAnswered === "boolean"
        ? data.sessionRecordingConsentAnswered
        : null,
    profile,
  };
}
