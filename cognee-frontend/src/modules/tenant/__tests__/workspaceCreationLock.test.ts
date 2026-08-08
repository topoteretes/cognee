import { tryAcquireWorkspaceCreationLock, releaseWorkspaceCreationLock } from "../workspaceCreationLock";

const LOCK_KEY = "cognee_workspace_creation_lock";

beforeEach(() => {
  localStorage.clear();
});

describe("workspaceCreationLock", () => {
  it("acquires the lock when none is held", () => {
    expect(tryAcquireWorkspaceCreationLock()).toBe(true);
    expect(localStorage.getItem(LOCK_KEY)).not.toBeNull();
  });

  it("fails to acquire when a fresh lock is already held", () => {
    localStorage.setItem(LOCK_KEY, String(Date.now()));
    expect(tryAcquireWorkspaceCreationLock()).toBe(false);
  });

  it("acquires the lock once a stale (expired) one is present", () => {
    localStorage.setItem(LOCK_KEY, String(Date.now() - 3 * 60 * 1000));
    expect(tryAcquireWorkspaceCreationLock()).toBe(true);
  });

  it("acquires the lock again after release", () => {
    expect(tryAcquireWorkspaceCreationLock()).toBe(true);
    releaseWorkspaceCreationLock();
    expect(localStorage.getItem(LOCK_KEY)).toBeNull();
    expect(tryAcquireWorkspaceCreationLock()).toBe(true);
  });

  it("treats a corrupted (non-numeric) lock value as stale", () => {
    localStorage.setItem(LOCK_KEY, "not-a-timestamp");
    expect(tryAcquireWorkspaceCreationLock()).toBe(true);
  });
});
