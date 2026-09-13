import type { RememberOptions } from "./rememberData";

/**
 * IndexedDB-backed record of an in-flight upload, so a refresh can reattach to
 * what is already server-side AND replay what never left the browser.
 *
 * localStorage cannot do this: it stores strings only, and the whole point is
 * keeping the pending File blobs. IndexedDB stores File/Blob natively via
 * structured clone, so the remainder of a selection survives a reload without
 * ever being re-read from disk by the user.
 */
export interface UploadSession {
  id: string;
  tenantId: string;
  datasetId: string;
  datasetName?: string;
  options?: RememberOptions;
  createdAt: number;
  updatedAt: number;
  filesTotal: number;
  bytesTotal: number;
  // Counts for what has already been accepted by the backend.
  filesCompleted: number;
  bytesCompleted: number;
  // The suffix of the original selection that has not been sent yet. Empty
  // once the upload leg finishes — the session then lives on only so polling
  // can be resumed.
  pending: File[];
  // Pipeline runs the backend acknowledged, one per landed batch; retained for
  // support/debugging of a session that later fails, since the run id is what a
  // backend log search needs.
  runIds: string[];
  stage: "uploading" | "processing";
  // Types of the original selection, for analytics that group by file type.
  // Kept on the session because a resumed run has no File objects for the
  // batches that already landed and would otherwise report an empty list.
  fileTypes?: string[];
  // When this session last actually moved — set on creation and refreshed
  // every time a batch lands. A failed upload keeps its session on purpose
  // (that is what makes a transient failure recoverable), so the bound on
  // retrying has to be "has it stopped progressing", not "how many times has a
  // page loaded": a slow upload that a user reloads past legitimately makes no
  // attempt-count progress while still moving forward.
  lastProgressAt?: number;
  // Set when the blobs could not be persisted (quota) — the session still
  // tracks progress and can reattach, it just cannot replay the remainder.
  degraded?: boolean;
  // False when NOTHING reached IndexedDB. The returned object then describes
  // only the in-memory run: it will not be found after a reload, so a caller
  // must not treat this session as durable.
  persisted?: boolean;
}

const DB_NAME = "cognee-uploads";
const DB_VERSION = 1;
const STORE = "sessions";
// Abandoned sessions (tab closed mid-upload and never resumed) would otherwise
// pin their blobs in IndexedDB forever.
const MAX_AGE_MS = 24 * 60 * 60 * 1000;

// Give up on a session that has stopped making progress for this long.
//
// Deliberately a stall window rather than an attempt count: counting page
// loads deletes a slow-but-healthy upload if the user happens to reload it a
// few times, which is the opposite of the intent. A session that keeps landing
// batches refreshes its clock and is never dropped; one that is genuinely
// wedged stops being picked up an hour after its last real progress, well
// before MAX_AGE_MS would expire it.
export const MAX_STALL_MS = 60 * 60 * 1000;

function available(): boolean {
  return typeof indexedDB !== "undefined";
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: "id" });
        store.createIndex("tenantId", "tenantId", { unique: false });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function run<T>(mode: IDBTransactionMode, work: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDb().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const tx = db.transaction(STORE, mode);
        const request = work(tx.objectStore(STORE));
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
        // Close on every terminal transaction state, not just success: an
        // aborted or errored transaction would otherwise leak the connection,
        // and a long upload writes once per batch.
        tx.oncomplete = () => db.close();
        tx.onabort = () => db.close();
        tx.onerror = () => db.close();
      }),
  );
}

/**
 * Every operation degrades to a no-op rather than throwing: persistence is a
 * resilience feature, and an upload must never fail because the browser is in
 * private mode, out of quota, or blocking storage.
 */
async function safely<T>(work: () => Promise<T>, fallback: T): Promise<T> {
  if (!available()) return fallback;
  try {
    return await work();
  } catch {
    return fallback;
  }
}

export function newSessionId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `upload-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export async function saveSession(session: UploadSession): Promise<UploadSession> {
  const record = { ...session, updatedAt: Date.now() };
  const stored = await safely(async () => {
    await run("readwrite", (store) => store.put(record));
    return true;
  }, false);

  if (stored) return { ...record, persisted: true };

  // Quota is the expected failure for a large selection. Retry once without the
  // blobs so the session still reattaches after a refresh — degraded, but the
  // already-uploaded files are not lost.
  //
  // Two flags come back from here and they mean different things:
  //
  // - `degraded` — something WAS written, but without the blobs. The record has
  //   an empty `pending` while files remain unsent, which is indistinguishable
  //   by shape from a session that finished uploading, so a resume that reads
  //   only `pending.length` would skip the upload phase and poll as though
  //   everything had landed. Callers MUST check it before treating an empty
  //   `pending` as "nothing left to send" (see useBrainUpload's resume path).
  // - `persisted: false` — NOTHING was written, neither the full record nor the
  //   metadata-only retry below. The returned object describes this run only;
  //   findResumableSession will never see it, so the session is ephemeral and
  //   will not survive a reload at all.
  if (record.pending.length > 0) {
    const metadataOnly = { ...record, pending: [], degraded: true, persisted: true };
    const ok = await safely(async () => {
      await run("readwrite", (store) => store.put(metadataOnly));
      return true;
    }, false);
    if (ok) return metadataOnly;
  }

  // Both writes failed (or IndexedDB is unavailable entirely). Returning
  // `record` here would hand the caller something that looks durable and is
  // not — findResumableSession will never see it. Report what is actually
  // true: nothing is replayable, and this session will not survive a reload.
  return { ...record, pending: [], degraded: true, persisted: false };
}

/**
 * Write a session only if it is still there.
 *
 * IndexedDB's `put` inserts when the key is absent, so a run that keeps saving
 * after another tab deleted its session would silently resurrect it — the
 * deleted record reappears and both tabs' sessions compete for the same
 * dataset. Progress updates during a run are updates, never creations, so they
 * go through here; only `upload()` creates, via `saveSession`.
 *
 * Returns null when the record is gone, which tells the caller it no longer
 * owns the run. A caller that keeps uploading after a null is making a
 * deliberate trade: the in-flight files still reach the backend, but the run
 * has no durable record from that point on, so it cannot be resumed if the tab
 * dies — the tab that took the dataset over owns the resumable session. Callers
 * should report the transition rather than swallow it (useBrainUpload does).
 */
export async function updateSession(session: UploadSession): Promise<UploadSession | null> {
  const record = { ...session, updatedAt: Date.now() };
  const ok = await safely(
    () =>
      openDb().then(
        (db) =>
          new Promise<boolean>((resolve, reject) => {
            const tx = db.transaction(STORE, "readwrite");
            const store = tx.objectStore(STORE);
            const existing = store.get(record.id);
            let found = false;
            existing.onsuccess = () => {
              if (!existing.result) return; // leave found false; tx completes as a no-op
              found = true;
              store.put(record);
            };
            existing.onerror = () => reject(existing.error);
            tx.oncomplete = () => {
              db.close();
              resolve(found);
            };
            tx.onabort = () => {
              db.close();
              reject(tx.error);
            };
            tx.onerror = () => {
              db.close();
              reject(tx.error);
            };
          }),
      ),
    false,
  );
  return ok ? record : null;
}

export async function deleteSession(id: string): Promise<void> {
  await safely(async () => {
    await run("readwrite", (store) => store.delete(id));
  }, undefined);
}

/**
 * Drop sessions past MAX_AGE_MS, for every tenant.
 *
 * Deliberately NOT scoped to the caller's tenant: an abandoned session pins its
 * file blobs in IndexedDB, and the tenant that abandoned it may never be opened
 * in this browser again. Scoping the sweep would leave those blobs on disk
 * forever. This is the one place that legitimately reads across tenants, and it
 * is a whole-store scan for that reason.
 */
export async function purgeExpiredSessions(): Promise<void> {
  const all = await safely(() => run<UploadSession[]>("readonly", (store) => store.getAll()), []);
  for (const session of all) {
    if (Date.now() - session.updatedAt > MAX_AGE_MS) await deleteSession(session.id);
  }
}

export async function listSessions(tenantId: string): Promise<UploadSession[]> {
  // Expiry first, then a read scoped by the tenantId index rather than a scan
  // filtered in JS — the index existed but nothing used it. Keeping the two
  // apart is also what lets the sweep stay global while the read stays
  // tenant-local; one query cannot be both.
  await purgeExpiredSessions();
  const mine = await safely(
    () => run<UploadSession[]>("readonly", (store) => store.index("tenantId").getAll(tenantId)),
    [],
  );
  // The sweep and this read are separate transactions, so a record can age out
  // between them. Cheap to re-check rather than hand back an expired session.
  return mine
    .filter((session) => Date.now() - session.updatedAt <= MAX_AGE_MS)
    .sort((a, b) => b.updatedAt - a.updatedAt);
}

/** The session a freshly loaded page should reattach to, if any. */
export async function findResumableSession(
  tenantId: string,
  datasetId?: string,
): Promise<UploadSession | null> {
  const sessions = await listSessions(tenantId);
  const match = datasetId ? sessions.filter((s) => s.datasetId === datasetId) : sessions;
  return match[0] ?? null;
}
