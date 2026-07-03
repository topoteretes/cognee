import { describe, expect, it } from "vitest";

import { parseOriginUrl } from "./git";

const GIT_CONFIG = [
  "[core]",
  "\trepositoryformatversion = 0",
  '[remote "origin"]',
  "\turl = https://github.com/user/repo.git",
  "\tfetch = +refs/heads/*:refs/remotes/origin/*",
  '[branch "main"]',
  "\tremote = origin",
].join("\n");

describe("parseOriginUrl", () => {
  it("extracts the origin remote url", () => {
    expect(parseOriginUrl(GIT_CONFIG)).toBe("https://github.com/user/repo.git");
  });

  it("returns null when there is no origin remote", () => {
    expect(parseOriginUrl("[core]\n\tbare = false\n")).toBeNull();
  });

  it("does not pick up a non-origin remote's url", () => {
    const config = ['[remote "upstream"]', "\turl = https://github.com/other/repo.git"].join("\n");
    expect(parseOriginUrl(config)).toBeNull();
  });
});
