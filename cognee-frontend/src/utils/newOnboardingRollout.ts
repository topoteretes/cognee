import { isRolloutEnabled } from "@/utils/isRolloutEnabled";

// Released hosts keep the current onboarding. Rolling the new one out to a
// host is deleting it from this set; once every host is out, the set, the
// hand-over and the old flow go together.
//
// Every cloud host is still here: the steps of the new onboarding are being
// built, so until they are all in, the new route runs on local dev and Vercel
// previews only.
const RELEASED_HOSTS: ReadonlySet<string> = new Set([
  "dev.cognee.ai",
  "staging.cognee.ai",
  "platform.cognee.ai",
]);

export function isNewOnboardingHost(host: string | null): boolean {
  return isRolloutEnabled(RELEASED_HOSTS, host);
}
