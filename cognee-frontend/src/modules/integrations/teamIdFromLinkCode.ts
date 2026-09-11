/** Segment count of a `/cognee-link` code: team, member, expiry, signature. */
const CODE_PARTS = 4;

/**
 * The Slack team id a `/cognee-link` code was issued for, or null if the code
 * is not shaped like one (CLO-390).
 *
 * Reading an unverified field for a *check*, never for a decision: the signature
 * can only be verified server-side, so this is used to warn that the active
 * workspace does not match the link, and the backend still validates the code
 * before acting on it.
 */
export function teamIdFromLinkCode(code: string): string | null {
  const parts = code.split(":");
  if (parts.length !== CODE_PARTS) return null;
  const [teamId] = parts;
  return teamId.length > 0 ? teamId : null;
}
