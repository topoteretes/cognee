import { renderHook } from "@testing-library/react";
import useSeedDatasetId from "../useSeedDatasetId";

const INSTANCE = "tenant-1";
const KEY = `cognee_last_graph_dataset:${INSTANCE}`;

interface Args {
  activeDatasetId?: string | null;
  setActiveDatasetId?: (id: string) => void;
  selectedDatasetId?: string | null;
  readableDatasetIds?: string[];
  readableLoading?: boolean;
  readableUnavailable?: boolean;
  governanceDatasetIds?: string[];
}

function args(overrides: Args, setActiveDatasetId: (id: string) => void) {
  return {
    instanceId: INSTANCE,
    activeDatasetId: overrides.activeDatasetId ?? null,
    setActiveDatasetId,
    selectedDatasetId: overrides.selectedDatasetId ?? null,
    readableDatasetIds: overrides.readableDatasetIds ?? [],
    readableLoading: overrides.readableLoading ?? false,
    readableUnavailable: overrides.readableUnavailable ?? false,
    governanceDatasetIds: overrides.governanceDatasetIds ?? [],
  };
}

function render(overrides: Args = {}) {
  const setActiveDatasetId = overrides.setActiveDatasetId ?? jest.fn();
  const view = renderHook((props: Args) => useSeedDatasetId(args(props, setActiveDatasetId)), {
    initialProps: overrides,
  });
  return { ...view, setActiveDatasetId };
}

describe("useSeedDatasetId", () => {
  afterEach(() => {
    localStorage.clear();
    jest.restoreAllMocks();
  });

  describe("opening", () => {
    it("opens on last visit's dataset on the very first render, before the list answers", () => {
      localStorage.setItem(KEY, "last-visit");
      const { setActiveDatasetId } = render({ readableLoading: true });
      expect(setActiveDatasetId).toHaveBeenCalledWith("last-visit");
    });

    it("opens on the first readable dataset when nothing was persisted", () => {
      const { setActiveDatasetId } = render({ readableDatasetIds: ["first-readable"] });
      expect(setActiveDatasetId).toHaveBeenCalledWith("first-readable");
    });

    it("does not reopen a dataset the list no longer carries", () => {
      localStorage.setItem(KEY, "deleted-since");
      const { setActiveDatasetId } = render({ readableDatasetIds: ["still-here"] });
      expect(setActiveDatasetId).toHaveBeenCalledWith("still-here");
      expect(setActiveDatasetId).not.toHaveBeenCalledWith("deleted-since");
    });

    it("leaves an already-open dataset alone", () => {
      const { setActiveDatasetId } = render({
        activeDatasetId: "already-open",
        readableDatasetIds: ["first-readable"],
      });
      expect(setActiveDatasetId).not.toHaveBeenCalled();
    });
  });

  describe("settling last visit's guess", () => {
    it("switches away once the list lands without it", () => {
      localStorage.setItem(KEY, "deleted-since");
      const { setActiveDatasetId, rerender } = render({
        readableLoading: true,
      });
      expect(setActiveDatasetId).toHaveBeenCalledWith("deleted-since");

      rerender({ activeDatasetId: "deleted-since", readableDatasetIds: ["still-here"] });
      expect(setActiveDatasetId).toHaveBeenCalledWith("still-here");
    });

    it("waits for the list rather than correcting mid-flight", () => {
      localStorage.setItem(KEY, "last-visit");
      const { setActiveDatasetId, rerender } = render({ readableLoading: true });
      (setActiveDatasetId as jest.Mock).mockClear();

      rerender({ activeDatasetId: "last-visit", readableLoading: true });
      expect(setActiveDatasetId).not.toHaveBeenCalled();
    });

    it("keeps it when the list confirms it", () => {
      localStorage.setItem(KEY, "last-visit");
      const { setActiveDatasetId, rerender } = render({ readableLoading: true });
      (setActiveDatasetId as jest.Mock).mockClear();

      rerender({ activeDatasetId: "last-visit", readableDatasetIds: ["last-visit", "other"] });
      expect(setActiveDatasetId).not.toHaveBeenCalled();
    });

    // Rendering once would pass this whatever the guard did: BusinessView
    // re-renders on every poll tick, hover and live event, so the honest
    // question is whether the guard survives the NEXT render, not the first.
    it("never pulls a brain the user picked themselves out from under them", () => {
      localStorage.setItem(KEY, "last-visit");
      const { setActiveDatasetId, rerender } = render({ readableLoading: true });
      (setActiveDatasetId as jest.Mock).mockClear();

      // The user picks an unreadable brain by hand, and the view keeps
      // rendering around them.
      rerender({ activeDatasetId: "user-picked", readableDatasetIds: ["still-here"] });
      rerender({ activeDatasetId: "user-picked", readableDatasetIds: ["still-here"] });
      rerender({ activeDatasetId: "user-picked", readableDatasetIds: ["still-here"] });
      expect(setActiveDatasetId).not.toHaveBeenCalled();
    });

    // Same shape as above: the write effect feeds activeDatasetId back into
    // storage, so anything comparing the live stored value against the active
    // one drifts into "correct every dataset, forever" after a render or two.
    it("stops correcting once the guess has been settled", () => {
      localStorage.setItem(KEY, "deleted-since");
      const { setActiveDatasetId, rerender } = render({ readableLoading: true });
      rerender({ activeDatasetId: "deleted-since", readableDatasetIds: ["still-here"] });
      (setActiveDatasetId as jest.Mock).mockClear();

      rerender({ activeDatasetId: "still-here", readableDatasetIds: ["still-here"] });
      rerender({ activeDatasetId: "still-here", readableDatasetIds: ["still-here"] });
      expect(setActiveDatasetId).not.toHaveBeenCalled();
    });

    it("leaves the dead id in place when there is nothing readable to move to", () => {
      localStorage.setItem(KEY, "deleted-since");
      const { setActiveDatasetId, rerender } = render({ readableLoading: true });
      (setActiveDatasetId as jest.Mock).mockClear();

      rerender({ activeDatasetId: "deleted-since", readableDatasetIds: [], readableUnavailable: true });
      expect(setActiveDatasetId).not.toHaveBeenCalled();
    });
  });

  describe("reporting the guess", () => {
    it("flags the opened dataset as unconfirmed while the list is still answering", () => {
      localStorage.setItem(KEY, "last-visit");
      const { result } = render({ readableLoading: true });
      expect(result.current.unconfirmedSeedId).toBe("last-visit");
    });

    it("clears the flag once the list vouches for it", () => {
      localStorage.setItem(KEY, "last-visit");
      const { result, rerender } = render({ readableLoading: true });

      rerender({ activeDatasetId: "last-visit", readableDatasetIds: ["last-visit"] });
      expect(result.current.unconfirmedSeedId).toBeNull();
    });

    it("never flags a dataset that came from the readable list itself", () => {
      const { result } = render({ readableDatasetIds: ["first-readable"] });
      expect(result.current.unconfirmedSeedId).toBeNull();
    });
  });

  describe("remembering", () => {
    it("records the dataset the graph ends up on", () => {
      render({ activeDatasetId: "opened", readableDatasetIds: ["opened"] });
      expect(localStorage.getItem(KEY)).toBe("opened");
    });

    it("does not record anything while no dataset is open", () => {
      render();
      expect(localStorage.getItem(KEY)).toBeNull();
    });

    it("survives storage being unavailable", () => {
      jest.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
        throw new Error("quota");
      });
      expect(() => render({ activeDatasetId: "opened" })).not.toThrow();
    });
  });
});
