import { act, renderHook, waitFor } from "@testing-library/react";

const mockShow = jest.fn();
jest.mock("@mantine/notifications", () => ({ notifications: { show: (...args: unknown[]) => mockShow(...args) } }));
jest.mock("@/utils/monitoring", () => ({ captureException: jest.fn() }));

import { useAssignmentChoice } from "../useAssignmentChoice";

interface Pending { choice: string | null; resolve: () => void; reject: () => void }

// Each save stays open until the test settles it, so overlapping saves can be
// finished in any order.
function setup() {
  const saves: Pending[] = [];
  const onChanged = jest.fn();
  const save = (choice: string | null) => new Promise<void>((resolve, reject) => {
    saves.push({ choice, resolve, reject: () => reject(new Error("pod unreachable")) });
  });
  const hook = renderHook(() => useAssignmentChoice({
    label: "Prompt", datasetId: "ds-1", datasetName: "Contracts", describe: (c) => c ?? "Automatic", save, onChanged,
  }));
  const settle = async (index: number, outcome: "resolve" | "reject") => { await act(async () => { saves[index][outcome](); }); };
  return { hook, saves, settle, onChanged };
}

beforeEach(() => jest.clearAllMocks());

describe("useAssignmentChoice", () => {
  it("shows a choice at once and keeps it once it is stored", async () => {
    const { hook, settle, onChanged } = setup();
    act(() => hook.result.current.setConfirmed("A"));

    act(() => hook.result.current.select("B"));
    expect(hook.result.current.selected).toBe("B");
    expect(onChanged).not.toHaveBeenCalled();

    await settle(0, "resolve");

    expect(hook.result.current.selected).toBe("B");
    expect(onChanged).toHaveBeenCalledTimes(1);
  });

  it("does nothing when the choice already showing is picked again", () => {
    const { hook, saves, onChanged } = setup();
    act(() => hook.result.current.setConfirmed("A"));

    act(() => hook.result.current.select("A"));

    expect(saves).toHaveLength(0);
    expect(onChanged).not.toHaveBeenCalled();
    expect(mockShow).not.toHaveBeenCalled();
  });

  // A configuration that loads after the user's pick overwrites what is showing
  // with what was stored, and a choice that is only showing was never stored.
  it("stores a choice that is showing but was never stored when it is picked", async () => {
    const { hook, saves, settle } = setup();
    act(() => hook.result.current.setConfirmed("A"));
    act(() => hook.result.current.select("B"));
    await settle(0, "reject");
    expect(hook.result.current.selected).toBe("A");

    act(() => hook.result.current.select("B"));

    expect(saves).toHaveLength(2);
  });

  it("goes back to the stored choice when the save fails", async () => {
    const { hook, settle, onChanged } = setup();
    act(() => hook.result.current.setConfirmed("A"));

    act(() => hook.result.current.select("B"));
    await settle(0, "reject");

    expect(hook.result.current.selected).toBe("A");
    expect(onChanged).not.toHaveBeenCalled();
    expect(mockShow).toHaveBeenCalledWith(expect.objectContaining({ title: "Prompt not confirmed", color: "red" }));
  });

  // The older save failing late must not undo what the user chose after it.
  it("leaves a newer choice alone when an older save fails", async () => {
    const { hook, settle } = setup();
    act(() => hook.result.current.setConfirmed("A"));

    act(() => hook.result.current.select("B"));
    act(() => hook.result.current.select("C"));
    await settle(0, "reject");

    expect(hook.result.current.selected).toBe("C");
  });

  // B then A, both failing: restoring "the value before the click" would leave
  // B showing, which was never stored.
  it("goes back to the stored choice, not the one before the click, after two failures in a row", async () => {
    const { hook, settle } = setup();
    act(() => hook.result.current.setConfirmed("A"));

    act(() => hook.result.current.select("B"));
    act(() => hook.result.current.select("A"));
    await settle(0, "reject");
    await settle(1, "reject");

    expect(hook.result.current.selected).toBe("A");
  });

  it("goes back to what the last successful save stored", async () => {
    const { hook, settle } = setup();
    act(() => hook.result.current.setConfirmed("A"));

    act(() => hook.result.current.select("B"));
    await settle(0, "resolve");
    act(() => hook.result.current.select("C"));
    await settle(1, "reject");

    expect(hook.result.current.selected).toBe("B");
  });

  // Double click on one option: one save, one toast, and the display follows
  // that save whichever way it ends.
  it("starts no second save for the choice that is already being saved", async () => {
    const { hook, saves, settle, onChanged } = setup();
    act(() => hook.result.current.setConfirmed("A"));

    act(() => hook.result.current.select("B"));
    act(() => hook.result.current.select("B"));
    await settle(0, "resolve");

    expect(saves).toHaveLength(1);
    expect(onChanged).toHaveBeenCalledTimes(1);
    expect(mockShow).toHaveBeenCalledTimes(1);
  });

  it("shows the stored choice after a double click whose single save fails", async () => {
    const { hook, settle } = setup();
    act(() => hook.result.current.setConfirmed("A"));

    act(() => hook.result.current.select("B"));
    act(() => hook.result.current.select("B"));
    await settle(0, "reject");

    expect(hook.result.current.selected).toBe("A");
  });

  // B, C, B again, with the first B failing: the user's last choice is B and it
  // is what ends up showing, not the A that was stored before any of them.
  it("keeps showing the newest choice when an earlier save of the same value fails", async () => {
    const { hook, settle } = setup();
    act(() => hook.result.current.setConfirmed("A"));

    act(() => hook.result.current.select("B"));
    act(() => hook.result.current.select("C"));
    act(() => hook.result.current.select("B"));
    await settle(0, "reject");
    await settle(1, "resolve");
    await settle(2, "resolve");

    expect(hook.result.current.selected).toBe("B");
  });

  it("picking the stored choice while another is saving stores it again, since that save may land last", async () => {
    const { hook, saves } = setup();
    act(() => hook.result.current.setConfirmed("A"));

    act(() => hook.result.current.select("B"));
    act(() => hook.result.current.select("A"));

    expect(saves.map((s) => s.choice)).toEqual(["B", "A"]);
  });

  // The page takes a new stored value (a reload of the config, a deleted item).
  // A choice saved before that must be savable again, not skipped as "already saved".
  it("saves a choice again after the stored value was replaced behind it", async () => {
    const { hook, saves, settle } = setup();
    act(() => hook.result.current.setConfirmed("A"));
    act(() => hook.result.current.select("B"));
    await settle(0, "resolve");

    act(() => hook.result.current.setConfirmed("A"));
    act(() => hook.result.current.select("B"));

    expect(saves).toHaveLength(2);
  });

  it("does not restore the display for a save that was superseded by a newly stored value", async () => {
    const { hook, settle } = setup();
    act(() => hook.result.current.setConfirmed("A"));
    act(() => hook.result.current.select("B"));

    act(() => hook.result.current.setConfirmed("X"));
    await settle(0, "reject");

    expect(hook.result.current.selected).toBe("X");
  });

  // A build started while a pick is still being stored would use a choice the
  // backend does not have yet.
  it("reports saving from a pick until its save has finished", async () => {
    const { hook, settle } = setup();
    act(() => hook.result.current.setConfirmed("A"));
    expect(hook.result.current.saving).toBe(false);

    act(() => hook.result.current.select("B"));
    expect(hook.result.current.saving).toBe(true);

    await settle(0, "resolve");
    expect(hook.result.current.saving).toBe(false);
  });

  it("stops saving after a failed save too", async () => {
    const { hook, settle } = setup();
    act(() => hook.result.current.setConfirmed("A"));

    act(() => hook.result.current.select("B"));
    await settle(0, "reject");

    expect(hook.result.current.saving).toBe(false);
  });

  // The doc promises "any save in flight": an older write can still land last.
  it("keeps saving until every overlapping save has finished, whichever order they end in", async () => {
    const { hook, settle } = setup();
    act(() => hook.result.current.setConfirmed("A"));

    act(() => hook.result.current.select("B"));
    act(() => hook.result.current.select("C"));
    await settle(1, "resolve");
    expect(hook.result.current.saving).toBe(true);

    await settle(0, "resolve");
    expect(hook.result.current.saving).toBe(false);
  });

  it("does not stay saving when the save throws before returning a promise", () => {
    const hook = renderHook(() => useAssignmentChoice({
      label: "Prompt", datasetId: "ds-1", datasetName: "Contracts", save: () => { throw new Error("sync failure"); }, onChanged: jest.fn(),
    }));

    act(() => hook.result.current.select("B"));

    return waitFor(() => expect(hook.result.current.saving).toBe(false));
  });

  // The page reads the configuration after it is up, and the user can pick first.
  describe("a configuration that arrives after the user has picked", () => {
    it("shows what was stored when nothing has been picked", () => {
      const { hook } = setup();

      act(() => hook.result.current.load("A"));

      expect(hook.result.current.selected).toBe("A");
    });

    it("leaves the pick showing", async () => {
      const { hook, settle } = setup();

      act(() => hook.result.current.select("B"));
      act(() => hook.result.current.load("A"));
      await settle(0, "resolve");

      expect(hook.result.current.selected).toBe("B");
    });

    it("goes back to what the configuration said if the pick then fails", async () => {
      const { hook, settle } = setup();

      act(() => hook.result.current.select("B"));
      act(() => hook.result.current.load("A"));
      await settle(0, "reject");

      expect(hook.result.current.selected).toBe("A");
    });

    // The pick is no longer showing once its save failed, so nothing the user
    // chose stands in the way of what is stored.
    it("shows what the configuration says when it arrives after a pick that failed", async () => {
      const { hook, settle } = setup();

      act(() => hook.result.current.select("B"));
      await settle(0, "reject");
      act(() => hook.result.current.load("A"));

      expect(hook.result.current.selected).toBe("A");
    });

    it("does not let a read taken before a confirmed save replace what that save stored", async () => {
      const { hook, settle } = setup();
      act(() => hook.result.current.load("A"));

      act(() => hook.result.current.select("B"));
      await settle(0, "resolve");
      act(() => hook.result.current.load("A"));
      act(() => hook.result.current.select("C"));
      await settle(1, "reject");

      expect(hook.result.current.selected).toBe("B");
    });

    it("lets the choice be picked again while its first save is still going and the configuration arrives", async () => {
      const { hook, saves, settle } = setup();

      act(() => hook.result.current.select("B"));
      act(() => hook.result.current.load("A"));
      act(() => hook.result.current.select("B"));

      expect(saves).toHaveLength(1);
      await settle(0, "resolve");
      expect(hook.result.current.selected).toBe("B");
    });
  });

  // A deleted item is taken as stored by other means. A save of it that was
  // already going must not bring it back.
  describe("a value stored by other means while a save is in flight", () => {
    it("is not overwritten by the save finishing", async () => {
      const { hook, settle } = setup();
      act(() => hook.result.current.setConfirmed("A"));

      act(() => hook.result.current.select("B"));
      act(() => hook.result.current.setConfirmed(null));
      await settle(0, "resolve");
      act(() => hook.result.current.select("C"));
      await settle(1, "reject");

      expect(hook.result.current.selected).toBeNull();
    });

    it("is not replaced when the save fails", async () => {
      const { hook, settle } = setup();
      act(() => hook.result.current.setConfirmed("A"));

      act(() => hook.result.current.select("B"));
      act(() => hook.result.current.setConfirmed(null));
      await settle(0, "reject");

      expect(hook.result.current.selected).toBeNull();
    });

    // The page deletes an item and takes it as unassigned; a configuration read
    // that was taken before the delete still names it.
    it("is not replaced by a configuration read taken before it", () => {
      const { hook } = setup();

      act(() => hook.result.current.setConfirmed(null));
      act(() => hook.result.current.load("A"));

      expect(hook.result.current.selected).toBeNull();
    });

    it("lets the same choice be picked and saved again", async () => {
      const { hook, saves, settle } = setup();
      act(() => hook.result.current.setConfirmed("A"));

      act(() => hook.result.current.select("B"));
      act(() => hook.result.current.setConfirmed("A"));
      act(() => hook.result.current.select("B"));
      await settle(0, "resolve");
      await settle(1, "resolve");

      expect(saves).toHaveLength(2);
      expect(hook.result.current.selected).toBe("B");
    });
  });
});
