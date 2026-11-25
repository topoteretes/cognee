import { isRolloutEnabled } from "@/utils/isRolloutEnabled";

const RELEASED: ReadonlySet<string> = new Set(["staging.cognee.ai", "platform.cognee.ai"]);

describe("isRolloutEnabled", () => {
  it.each(["dev.cognee.ai", "cognee-saas-git-feature-x-topoteretes.vercel.app", "localhost"])(
    "enables the feature on %s",
    (host) => {
      expect(isRolloutEnabled(RELEASED, host)).toBe(true);
    },
  );

  it.each(["staging.cognee.ai", "platform.cognee.ai"])("disables the feature on the released host %s", (host) => {
    expect(isRolloutEnabled(RELEASED, host)).toBe(false);
  });

  it("ignores the port a server-side host header carries", () => {
    expect(isRolloutEnabled(RELEASED, "localhost:3000")).toBe(true);
    expect(isRolloutEnabled(RELEASED, "platform.cognee.ai:443")).toBe(false);
  });

  it("ignores the case of the host", () => {
    expect(isRolloutEnabled(RELEASED, "Platform.Cognee.AI")).toBe(false);
  });

  it("ignores a trailing dot, which names the same host", () => {
    expect(isRolloutEnabled(RELEASED, "platform.cognee.ai.")).toBe(false);
    expect(isRolloutEnabled(RELEASED, "dev.cognee.ai.")).toBe(true);
  });

  it.each([
    null,
    "",
    "dev.cognee.ai, platform.cognee.ai",
    "dev.cognee.ai,platform.cognee.ai",
    "platform.cognee.ai@evil.com",
    "platform.cognee.ai/path",
    "not a host",
  ])(
    "treats %j as a host that is not known and disables the feature",
    (host) => {
      expect(isRolloutEnabled(RELEASED, host)).toBe(false);
    },
  );

  it("enables the feature everywhere once the released list is empty, except for an unknown host", () => {
    expect(isRolloutEnabled(new Set(), "platform.cognee.ai")).toBe(true);
    expect(isRolloutEnabled(new Set(), null)).toBe(false);
  });
});
