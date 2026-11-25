/**
 * Tells the browser's other tabs that session-recording consent was withdrawn.
 *
 * Withdrawal used to be one tab deep. PrivacyWidget stops Sentry's replay and
 * reloads, which removes the Contentsquare tag in the tab the toggle was used
 * in — but a second tab opened while consent still stood keeps its own tag and
 * its own replay until its `/api/me` query happens to refetch. The server-side
 * record is authoritative from the next load, so it self-heals eventually;
 * "eventually" is not what a withdrawal means, and it is the tab the user is
 * NOT looking at that lasts longest.
 *
 * A `storage` event is the transport (the pattern lastGraphDataset already
 * uses) because it needs no permission, fires only in the OTHER tabs by
 * definition, and degrades to the pre-existing self-heal if storage is
 * unavailable. The value is a timestamp so two withdrawals in a row still
 * count as a change and still fire.
 *
 * Deliberately one-directional: withdrawal travels, a grant does not. Starting
 * a recording in a tab on the strength of a message from another one would put
 * the decision to record somewhere other than the consent answer this tab has
 * read, which is the one thing the gate exists to prevent. A grant waits for
 * that tab's own /api/me.
 */

// Account-scoped by value rather than by key, and not registered in
// sessionScopedStorage: this is a signal, not state. It carries the user it
// was written for so a withdrawal by one account cannot stop a recording a
// different account consented to in another tab.
const KEY = "cognee_session_recording_withdrawn";

interface WithdrawalSignal {
  userId: string | null;
  at: number;
}

export function broadcastSessionRecordingWithdrawal(userId: string | null): void {
  if (typeof window === "undefined") return;
  try {
    const signal: WithdrawalSignal = { userId, at: Date.now() };
    window.localStorage.setItem(KEY, JSON.stringify(signal));
  } catch {
    // Storage can be unavailable (private mode, quota, a blocking policy).
    // The other tabs then fall back to correcting themselves on their next
    // load, which is where this started — never a reason to fail the
    // withdrawal the user actually asked for.
  }
}

export function subscribeToSessionRecordingWithdrawal(
  userId: string | null,
  onWithdrawn: () => void,
): () => void {
  if (typeof window === "undefined") return () => {};
  const onStorage = (event: StorageEvent): void => {
    if (event.key !== KEY || !event.newValue) return;
    let signal: WithdrawalSignal;
    try {
      signal = JSON.parse(event.newValue) as WithdrawalSignal;
    } catch {
      return;
    }
    // An anonymous signal is ignored rather than applied to everyone: with no
    // user to compare, "stop recording" and "stop someone else's recording"
    // are the same message.
    if (signal.userId === null || signal.userId !== userId) return;
    onWithdrawn();
  };
  window.addEventListener("storage", onStorage);
  return () => window.removeEventListener("storage", onStorage);
}
