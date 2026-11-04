/**
 * Session-recording consent vocabulary for the open-source build.
 *
 * The consent itself is SaaS-only: it gates Contentsquare and Sentry Session
 * Replay, neither of which exists here, and the dialog and settings toggle
 * that collect it are excluded from the sync. What survives is the source
 * vocabulary, because UserContext and UserProvider are synced and type the
 * consent setter with it.
 */
export const SESSION_RECORDING_CONSENT_SOURCES = ["first_login", "settings"] as const;

export type SessionRecordingConsentSource = (typeof SESSION_RECORDING_CONSENT_SOURCES)[number];

export function isSessionRecordingConsentSource(value: unknown): value is SessionRecordingConsentSource {
  return SESSION_RECORDING_CONSENT_SOURCES.includes(value as SessionRecordingConsentSource);
}
