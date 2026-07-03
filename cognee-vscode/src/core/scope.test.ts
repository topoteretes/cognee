import { describe, expect, it } from "vitest";

import { deriveDatasetName, normalizeGitRemote, sanitizeDatasetName } from "./scope";

describe("normalizeGitRemote", () => {
  it("normalizes ssh, https, and scheme+userinfo forms to the same key", () => {
    expect(normalizeGitRemote("git@github.com:user/repo.git")).toBe("github.com/user/repo");
    expect(normalizeGitRemote("https://github.com/user/repo.git")).toBe("github.com/user/repo");
    expect(normalizeGitRemote("ssh://git@github.com/user/repo/")).toBe("github.com/user/repo");
    expect(normalizeGitRemote("HTTPS://GitHub.com/User/Repo")).toBe("github.com/user/repo");
  });

  it("returns null for empty or missing input", () => {
    expect(normalizeGitRemote("")).toBeNull();
    expect(normalizeGitRemote("   ")).toBeNull();
    expect(normalizeGitRemote(null)).toBeNull();
    expect(normalizeGitRemote(undefined)).toBeNull();
  });
});

describe("sanitizeDatasetName", () => {
  it("replaces disallowed characters and trims separators", () => {
    expect(sanitizeDatasetName("My Project!")).toBe("My_Project");
    expect(sanitizeDatasetName("  a//b  ")).toBe("a_b");
    expect(sanitizeDatasetName("!!!")).toBe("workspace");
  });
});

describe("deriveDatasetName", () => {
  it("is deterministic and identical across equivalent remotes", () => {
    const fromSsh = deriveDatasetName({ workspaceRoot: "/a", gitRemote: "git@github.com:user/repo.git" });
    const fromHttps = deriveDatasetName({
      workspaceRoot: "/b",
      gitRemote: "https://github.com/user/repo.git",
    });
    expect(fromSsh).toBe(fromHttps);
    expect(fromSsh).toMatch(/^vscode_[0-9a-f]{16}$/);
  });

  it("falls back to the workspace path (separator/trailing-slash insensitive)", () => {
    const a = deriveDatasetName({ workspaceRoot: "/home/x/proj" });
    const b = deriveDatasetName({ workspaceRoot: "/home/x/proj/" });
    const c = deriveDatasetName({ workspaceRoot: "\\home\\x\\proj" });
    expect(a).toBe(b);
    expect(a).toBe(c);
    expect(a).toMatch(/^vscode_[0-9a-f]{16}$/);
  });

  it("honors an explicit override", () => {
    expect(deriveDatasetName({ workspaceRoot: "/a", override: "My Project!" })).toBe("My_Project");
  });

  it("respects a custom prefix", () => {
    expect(deriveDatasetName({ workspaceRoot: "/a", prefix: "jetbrains" })).toMatch(
      /^jetbrains_[0-9a-f]{16}$/,
    );
  });
});
