import pickSeedDatasetId from "../pickSeedDatasetId";

const base = {
  selectedDatasetId: null,
  persistedDatasetId: null,
  readableDatasetIds: [],
  readableLoading: false,
  governanceDatasetIds: [],
  readableUnavailable: false,
};

describe("pickSeedDatasetId", () => {
  it("opens on the dataset the user already picked in the filter", () => {
    const id = pickSeedDatasetId({
      ...base,
      selectedDatasetId: "picked",
      persistedDatasetId: "last-visit",
      readableDatasetIds: ["first-readable"],
    });
    expect(id).toBe("picked");
  });

  it("falls back to the first readable dataset when nothing is picked", () => {
    const id = pickSeedDatasetId({ ...base, readableDatasetIds: ["first-readable", "second"] });
    expect(id).toBe("first-readable");
  });

  it("ignores governance while the readable list is available", () => {
    const id = pickSeedDatasetId({ ...base, governanceDatasetIds: ["tenant-only"] });
    expect(id).toBeNull();
  });

  it("uses governance only when the readable list could not be fetched", () => {
    const id = pickSeedDatasetId({
      ...base,
      governanceDatasetIds: ["tenant-only"],
      readableUnavailable: true,
    });
    expect(id).toBe("tenant-only");
  });

  it("returns null when there is nothing to open", () => {
    expect(pickSeedDatasetId({ ...base, readableUnavailable: true })).toBeNull();
  });

  describe("last visit's dataset", () => {
    it("opens on it while the readable list is still in flight", () => {
      const id = pickSeedDatasetId({ ...base, persistedDatasetId: "last-visit", readableLoading: true });
      expect(id).toBe("last-visit");
    });

    it("keeps it once the list confirms it is still readable", () => {
      const id = pickSeedDatasetId({
        ...base,
        persistedDatasetId: "last-visit",
        readableDatasetIds: ["first-readable", "last-visit"],
      });
      expect(id).toBe("last-visit");
    });

    it("drops it for the first readable one when the list no longer carries it", () => {
      const id = pickSeedDatasetId({
        ...base,
        persistedDatasetId: "deleted-since",
        readableDatasetIds: ["first-readable"],
      });
      expect(id).toBe("first-readable");
    });

    // The distinction the earlier version could not make: an empty workspace
    // and a list still in flight both arrive as an empty array, and reopening
    // a deleted dataset on a workspace with nothing left put the user on a
    // connection error they could not act on.
    it("is not reused on a workspace the list reports as genuinely empty", () => {
      const id = pickSeedDatasetId({ ...base, persistedDatasetId: "deleted-since" });
      expect(id).toBeNull();
    });

    it("is preferred over governance when the readable list could not be fetched", () => {
      const id = pickSeedDatasetId({
        ...base,
        persistedDatasetId: "last-visit",
        governanceDatasetIds: ["tenant-only"],
        readableUnavailable: true,
      });
      expect(id).toBe("last-visit");
    });
  });
});
