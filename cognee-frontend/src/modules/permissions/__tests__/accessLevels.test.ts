import { accessLevelLabel, levelFor, permissionsFor } from "@/modules/permissions/accessLevels";

describe("accessLevels", () => {
  describe("levelFor", () => {
    it("reads read alone as view", () => {
      expect(levelFor(["read"])).toBe("view");
    });

    it("reads read and write as edit", () => {
      expect(levelFor(["write", "read"])).toBe("edit");
    });

    it("reads all four permissions as full access", () => {
      expect(levelFor(["share", "delete", "write", "read"])).toBe("full");
    });

    it("returns null for read+share, which matches no level's bundle exactly", () => {
      expect(levelFor(["read", "share"])).toBeNull();
    });

    it("returns null for read+write+share, since it holds share beyond edit but not delete", () => {
      expect(levelFor(["read", "write", "share"])).toBeNull();
    });

    it("returns null for read+write+delete, since full access also requires share", () => {
      expect(levelFor(["read", "write", "delete"])).toBeNull();
    });

    it("returns null when not even read is held", () => {
      expect(levelFor(["write", "delete"])).toBeNull();
    });
  });

  describe("permissionsFor", () => {
    it("bundles each level's permissions", () => {
      expect(permissionsFor("view")).toEqual(["read"]);
      expect(permissionsFor("edit")).toEqual(["read", "write"]);
      expect(permissionsFor("full")).toEqual(["read", "write", "delete", "share"]);
    });

    it("returns an empty bundle for a level that isn't one of the three", () => {
      expect(permissionsFor("unknown" as never)).toEqual([]);
    });
  });

  describe("accessLevelLabel", () => {
    it("labels each level", () => {
      expect(accessLevelLabel("view")).toBe("Can view");
      expect(accessLevelLabel("edit")).toBe("Can edit");
      expect(accessLevelLabel("full")).toBe("Full access");
    });

    it("falls back to the raw value for a level that isn't one of the three", () => {
      expect(accessLevelLabel("unknown" as never)).toBe("unknown");
    });
  });
});
