/**
 * The OAuth callback's outcome, carried from the popup back to the page that
 * opened it.
 *
 * The control plane ends every install round-trip by redirecting the popup to
 * `/integrations?<provider>=<outcome>` (see `_frontend_redirect` in
 * cognee-saas-backend). The popup closes itself and the opener notices via its
 * poll — but the outcome lives in the popup's URL, which the opener can never
 * read: by the time the poll fires, the window is gone.
 *
 * So the popup writes it down before closing and the opener picks it up. A
 * one-slot localStorage handoff rather than postMessage, matching the choice
 * already made in `useConnectorConnect` — both windows are same-origin, and
 * this keeps the popup's only job "record and close".
 *
 * Until this existed the outcome was simply dropped, so a refused install was
 * indistinguishable from a successful one: popup closes, status refetches,
 * modal falls back to its initial state with nothing said. `error_already_
 * connected` was the worst of them — no amount of retrying can ever fix it,
 * and retrying was the only thing the UI left you.
 */

const STORAGE_KEY = "cognee-oauth-outcome";

/**
 * Long enough to survive the popup close plus the opener's 600 ms poll, short
 * enough that an outcome nobody consumed (opener navigated away, tab crashed)
 * can never surface against an unrelated attempt later.
 */
const MAX_AGE_MS = 2 * 60 * 1000;

/** Outcomes `_frontend_redirect` can send. Anything else is handled generically. */
export type OAuthOutcome =
  | "connected"
  | "cancelled"
  | "error_invalid_state"
  | "error_already_connected"
  | "error_exchange_failed"
  // Framework connectors (CLO-637) add three of their own.
  | "error_not_configured"
  | "error_not_authorized"
  | "error_session_expired"
  | "error_start_failed"
  /**
   * Not a failure: the control plane wants the owner to pick which account or
   * org to link (CLO-627). Always carries a `choiceToken` alongside it — see
   * `takeOAuthOutcome`.
   */
  | "choose_installation";

export interface OAuthFailure {
  provider: string;
  outcome: string;
}

/** What `takeOAuthOutcome` hands back once a popup has reported in. */
export interface TakenOutcome {
  outcome: string;
  /** Only ever set for `choose_installation` — null for every other outcome. */
  choiceToken: string | null;
}

interface StoredOutcome {
  provider: string;
  outcome: string;
  choiceToken?: string;
  at: number;
}

/**
 * Record the outcome from inside the popup, immediately before it closes.
 * `choiceToken` rides the same one-slot handoff as the outcome itself
 * (CLO-627's `choose_installation`) — it is never given a URL or a storage
 * slot of its own, so it is taken-and-cleared exactly once, the same as every
 * other outcome, and the opener's own address bar never holds it.
 */
export function stashOAuthOutcome(provider: string, outcome: string | null, choiceToken: string | null = null): void {
  if (!outcome) return;
  try {
    const payload: StoredOutcome = {
      provider,
      outcome,
      at: Date.now(),
      ...(choiceToken ? { choiceToken } : {}),
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // Private mode, quota, storage disabled — the connect flow itself still
    // works, the user just loses the explanation. Never break the popup close.
  }
}

/**
 * Read and clear the outcome for `provider`, if the popup left one.
 *
 * Consuming it here is what stops a failure from being re-reported on the next
 * attempt. Returns null when the popup was dismissed by hand (it never reached
 * the callback, so it never wrote one) — which must stay silent, because the
 * user closing the window is not an error.
 */
export function takeOAuthOutcome(provider: string): TakenOutcome | null {
  let raw: string | null = null;
  try {
    raw = window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
  if (!raw) return null;

  let stored: StoredOutcome;
  try {
    stored = JSON.parse(raw) as StoredOutcome;
  } catch {
    return null;
  }

  // Checked before the key is ever removed: this is the one slot shared by
  // every connector, and a poll for one provider firing before another's
  // popup has closed is a real sequence (open GitHub, close its modal without
  // cancelling the poll, connect Slack instead — both intervals are now live
  // against the same slot). Clearing first and checking after meant whichever
  // poll ran first consumed and destroyed an outcome that was never its own,
  // and the provider it actually belonged to reported nothing at all
  // (review-reported) — exactly the silent failure this module exists to
  // prevent.
  if (stored.provider !== provider) return null;

  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Read succeeded, the clear didn't — the outcome is still returned below;
    // a stale entry lingering past its removal is what MAX_AGE_MS is for.
  }

  if (!Number.isFinite(stored.at) || Date.now() - stored.at > MAX_AGE_MS) return null;
  if (typeof stored.outcome !== "string") return null;
  return { outcome: stored.outcome, choiceToken: stored.choiceToken ?? null };
}

/**
 * Whether an outcome means the install did not happen.
 *
 * A type guard so callers get the narrowed string without a cast — and so a
 * missing outcome (hand-closed popup) can only ever take the silent path.
 * `choose_installation` is the one other non-failure: the install is still
 * pending the owner's pick, not refused.
 */
export function isFailureOutcome(outcome: string | null): outcome is string {
  return outcome !== null && outcome !== "connected" && outcome !== "choose_installation";
}

/**
 * The sentence shown in the modal for a failed install.
 *
 * Each one names what actually happened and what the reader can do about it —
 * a generic "something went wrong" would leave `error_already_connected`
 * looking retryable, which is the single most misleading thing this UI can say.
 * The slug is appended for anything unrecognised so a new backend outcome
 * degrades to something still reportable rather than to silence.
 */
export function describeOAuthFailure(outcome: string, providerName: string): string {
  switch (outcome) {
    case "cancelled":
      return `Authorization was cancelled, so ${providerName} is not connected. You can start again whenever you like.`;
    case "error_already_connected":
      return `This ${providerName} workspace is already connected to a different Cognee workspace. A ${providerName} workspace can only be connected in one place — disconnect it there first, then connect it here.`;
    case "error_invalid_state":
      return `The authorization link expired before it was approved. Close this and start again — the link is only valid for a few minutes.`;
    case "error_exchange_failed":
      return `${providerName} refused the authorization. Try once more; if it keeps failing, this needs a look at the ${providerName} app configuration.`;
    case "error_not_configured":
      return `${providerName} is not switched on in this environment yet, so the authorization could not be completed.`;
    case "error_start_failed":
      // The provider was never reached: this is the authorize call itself
      // failing, so nothing about the provider's own configuration is implied.
      return `Could not start connecting ${providerName}. Please try again.`;
    case "error_session_expired":
      // An org owner's approval can take longer than a session lasts, and the
      // completing call is authenticated — so this is the one failure the
      // provider had no part in, and telling the reader to go audit the app
      // configuration would send them somewhere with nothing to find.
      return `Your session expired while ${providerName} was being authorized, so the connection was not saved. Sign in again and reconnect.`;
    case "error_not_authorized":
      // The control plane re-checks ownership when the browser comes back,
      // because an install takes long enough for it to have changed since.
      return `You are no longer the owner of this workspace, so the ${providerName} connection was not saved. Ask the current owner to connect it.`;
    default:
      return `Connecting ${providerName} failed (${outcome}).`;
  }
}
