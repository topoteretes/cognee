import { ConsentedSessionReplay, IdentifyUser } from "@/modules/analytics";
import isCloudEnvironment from "@/utils/isCloudEnvironment";
import type { ReactNode } from "react";

// IdentifyUser is mounted here as well as in (app), because this group is where
// a brand-new account actually lands: the (app) layout sends anyone with
// isSeenWelcome === false to /welcome before any (app) page paints. Without
// IdentifyUser here, the sign-up consent flush and the `Signed Up` event both
// waited until the user finished onboarding and reached an (app) page, so
// anyone who dropped out before then was never recorded at all (CLO-781).
//
// Mounting it twice needs a guard, and the per-path ones are not enough: the
// mount that fires via the intent cookie never reaches the logins_count
// fallback, so the fallback's own flag stays unset and the second mount
// happily emits a duplicate `Signed Up`. IdentifyUser's cognee_signup_tracked
// flag is what makes the second mount a no-op.
export default function SetupLayout({ children }: { children: ReactNode }) {
  return (
    <>
      {isCloudEnvironment() && <IdentifyUser />}
      {isCloudEnvironment() && <ConsentedSessionReplay />}
      {children}
    </>
  );
}
