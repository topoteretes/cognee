import { describe, expect, it } from "vitest";

import { PathIndex, type KeyValueStore } from "./pathIndex";

function fakeStore(): KeyValueStore {
  const data = new Map<string, unknown>();
  return {
    get<T>(key: string): T | undefined {
      return data.get(key) as T | undefined;
    },
    update(key: string, value: unknown): Thenable<void> {
      data.set(key, value);
      return Promise.resolve();
    },
  };
}

describe("PathIndex", () => {
  it("records and resolves an exact path by basename", async () => {
    const index = new PathIndex(fakeStore());
    await index.record("themes/_welcome/composer.json");
    expect(index.pathsFor("composer.json")).toEqual(["themes/_welcome/composer.json"]);
  });

  it("keeps distinct paths for the same basename, sorted and de-duplicated", async () => {
    const index = new PathIndex(fakeStore());
    await index.record("themes/_welcome/composer.json");
    await index.record("themes/attract/composer.json");
    await index.record("themes/_welcome/composer.json"); // duplicate ignored
    expect(index.pathsFor("composer.json")).toEqual([
      "themes/_welcome/composer.json",
      "themes/attract/composer.json",
    ]);
  });

  it("normalizes separators and leading ./ when recording", async () => {
    const index = new PathIndex(fakeStore());
    await index.record(".\\src\\app\\main.ts");
    expect(index.pathsFor("main.ts")).toEqual(["src/app/main.ts"]);
  });

  it("falls back to the stem when the cited name has no extension", async () => {
    const index = new PathIndex(fakeStore());
    await index.record("src/app/main.ts");
    expect(index.pathsFor("main")).toEqual(["src/app/main.ts"]);
  });

  it("returns an empty list for unknown or empty names", async () => {
    const index = new PathIndex(fakeStore());
    await index.record("   ");
    expect(index.pathsFor("nope.txt")).toEqual([]);
    expect(index.pathsFor("")).toEqual([]);
  });
});
