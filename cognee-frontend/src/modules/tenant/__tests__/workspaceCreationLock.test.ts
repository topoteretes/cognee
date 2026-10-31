import {
  isWorkspaceCreationLockHeld,
  tryAcquireWorkspaceCreationLock,
  releaseWorkspaceCreationLock,
} from "../workspaceCreationLock";

const LOCK_KEY = "cognee_workspace_creation_lock";

beforeEach(() => {
  localStorage.clear();
});

describe("workspaceCreationLock", () => {
  it("acquires the lock when none is held", () => {
    expect(tryAcquireWorkspaceCreationLock()).not.toBeNull();
    expect(localStorage.getItem(LOCK_KEY)).not.toBeNull();
  });

  it("fails to acquire when a fresh lock is already held", () => {
    tryAcquireWorkspaceCreationLock();
    expect(tryAcquireWorkspaceCreationLock()).toBeNull();
  });

  it("acquires the lock once a stale (expired) one is present", () => {
    localStorage.setItem(LOCK_KEY, `someone:${Date.now() - 3 * 60 * 1000}`);
    expect(tryAcquireWorkspaceCreationLock()).not.toBeNull();
  });

  it("acquires the lock again after release", () => {
    const holder = tryAcquireWorkspaceCreationLock();
    expect(holder).not.toBeNull();
    releaseWorkspaceCreationLock(holder ?? "");
    expect(localStorage.getItem(LOCK_KEY)).toBeNull();
    expect(tryAcquireWorkspaceCreationLock()).not.toBeNull();
  });

  it("treats a corrupted (non-numeric) lock value as stale", () => {
    localStorage.setItem(LOCK_KEY, "not-a-timestamp");
    expect(tryAcquireWorkspaceCreationLock()).not.toBeNull();
  });

  it("honors a fresh lock written in the pre-holder-id format", () => {
    // Another tab still running the previous build during a rollout.
    localStorage.setItem(LOCK_KEY, String(Date.now()));
    expect(tryAcquireWorkspaceCreationLock()).toBeNull();
  });

  it("does not let a late holder release a lock someone else now owns", () => {
    // A poll loop can outlive the lock's TTL. Without holder identity, its
    // release would clear the lock a second tab had since acquired, reopening
    // the duplicate-create window the lock exists to close.
    const stale = tryAcquireWorkspaceCreationLock() ?? "";
    localStorage.setItem(LOCK_KEY, `other-tab:${Date.now()}`);

    releaseWorkspaceCreationLock(stale);

    expect(localStorage.getItem(LOCK_KEY)).toBe(`other-tab:${localStorage.getItem(LOCK_KEY)?.split(":")[1]}`);
    expect(isWorkspaceCreationLockHeld()).toBe(true);
  });

  it("reports a lock as no longer held once it expires", () => {
    localStorage.setItem(LOCK_KEY, `someone:${Date.now() - 3 * 60 * 1000}`);
    expect(isWorkspaceCreationLockHeld()).toBe(false);
  });
});
