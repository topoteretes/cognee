import {
  broadcastSessionRecordingWithdrawal,
  subscribeToSessionRecordingWithdrawal,
} from "../sessionRecordingWithdrawal";

const KEY = "cognee_session_recording_withdrawn";

// A `storage` event never fires in the tab that wrote the value, so the other
// tab has to be simulated rather than produced by calling the broadcaster.
function storageEventFrom(key: string, newValue: string | null): void {
  window.dispatchEvent(new StorageEvent("storage", { key, newValue }));
}

afterEach(() => {
  window.localStorage.clear();
});

describe("broadcastSessionRecordingWithdrawal", () => {
  it("writes the withdrawing user, so other tabs can tell whose answer changed", () => {
    broadcastSessionRecordingWithdrawal("auth0|1");

    expect(JSON.parse(window.localStorage.getItem(KEY) as string)).toMatchObject({ userId: "auth0|1" });
  });

  it("changes the value on a repeat withdrawal", () => {
    // An identical value fires no storage event, so a second withdrawal after
    // a re-grant would reach no one.
    broadcastSessionRecordingWithdrawal("auth0|1");
    const first = window.localStorage.getItem(KEY);
    jest.spyOn(Date, "now").mockReturnValue(Date.now() + 1000);

    broadcastSessionRecordingWithdrawal("auth0|1");

    expect(window.localStorage.getItem(KEY)).not.toBe(first);
    jest.restoreAllMocks();
  });

  it("does not throw when storage is unavailable", () => {
    // Private mode and blocking policies throw on write. The withdrawal the
    // user asked for must still complete; the other tabs fall back to
    // correcting themselves on their next load.
    jest.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });

    expect(() => broadcastSessionRecordingWithdrawal("auth0|1")).not.toThrow();
    jest.restoreAllMocks();
  });
});

describe("subscribeToSessionRecordingWithdrawal", () => {
  it("fires when the same account withdraws in another tab", () => {
    const onWithdrawn = jest.fn();
    subscribeToSessionRecordingWithdrawal("auth0|1", onWithdrawn);

    storageEventFrom(KEY, JSON.stringify({ userId: "auth0|1", at: 1 }));

    expect(onWithdrawn).toHaveBeenCalledTimes(1);
  });

  it("ignores a withdrawal by a different account", () => {
    // Two accounts can share a browser. Stopping a recording the user in this
    // tab consented to, because someone else withdrew theirs, is a different
    // bug in the same place.
    const onWithdrawn = jest.fn();
    subscribeToSessionRecordingWithdrawal("auth0|1", onWithdrawn);

    storageEventFrom(KEY, JSON.stringify({ userId: "auth0|2", at: 1 }));

    expect(onWithdrawn).not.toHaveBeenCalled();
  });

  it("ignores an anonymous signal", () => {
    const onWithdrawn = jest.fn();
    subscribeToSessionRecordingWithdrawal("auth0|1", onWithdrawn);

    storageEventFrom(KEY, JSON.stringify({ userId: null, at: 1 }));

    expect(onWithdrawn).not.toHaveBeenCalled();
  });

  it("ignores writes to other keys and unreadable values", () => {
    const onWithdrawn = jest.fn();
    subscribeToSessionRecordingWithdrawal("auth0|1", onWithdrawn);

    storageEventFrom("cognee_selected_tenant", JSON.stringify({ userId: "auth0|1", at: 1 }));
    storageEventFrom(KEY, "not json");
    storageEventFrom(KEY, null);

    expect(onWithdrawn).not.toHaveBeenCalled();
  });

  it("stops listening once unsubscribed", () => {
    const onWithdrawn = jest.fn();
    const unsubscribe = subscribeToSessionRecordingWithdrawal("auth0|1", onWithdrawn);

    unsubscribe();
    storageEventFrom(KEY, JSON.stringify({ userId: "auth0|1", at: 1 }));

    expect(onWithdrawn).not.toHaveBeenCalled();
  });
});
