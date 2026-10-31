import { isNewOnboardingHost } from "@/utils/newOnboardingRollout";

describe("isNewOnboardingHost", () => {
  it.each(["cognee-saas-git-x-topoteretes.vercel.app", "localhost:3000"])(
    "shows the new onboarding on %s",
    (host) => {
      expect(isNewOnboardingHost(host)).toBe(true);
    },
  );

  it.each(["dev.cognee.ai", "staging.cognee.ai", "platform.cognee.ai", null])(
    "keeps the current onboarding on %s",
    (host) => {
      expect(isNewOnboardingHost(host)).toBe(false);
    },
  );
});
