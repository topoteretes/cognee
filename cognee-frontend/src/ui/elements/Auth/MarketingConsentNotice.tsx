"use client";

import { Text } from "@mantine/core";

// The policy hosted on the marketing site, which is also what its own footer
// links to. Not the in-app /privacy-policy route: that one lives inside the
// authenticated (app) group and is therefore unreachable from the sign-up page.
const PRIVACY_POLICY_URL = "https://www.cognee.ai/privacy-notice";

/**
 * Marketing-email notice, shown on both the sign-up and sign-in forms
 * (CLO-781/670).
 *
 * Not a checkbox. The earlier version made every email sign-up conditional
 * on an affirmative tick, which is presumed not freely given under GDPR
 * Art. 7(4)/Recital 43 once the account itself depends on the answer — and
 * Google/GitHub were never gated the same way, so the requirement did not
 * even achieve coverage. Creating an account is now the consent, stated
 * plainly rather than gated behind a control: every sign-up parks
 * `optedIn: true` regardless of anything on this page (see
 * signupWithPassword.ts / recordSignupConsent.ts). This text is the
 * disclosure that makes that true, not a form control — there is nothing
 * here to check, submit, or silently leave unticked.
 *
 * Shown on Sign In too, not only Sign Up: Auth0 does not distinguish the two
 * for Google/GitHub, so a never-before-seen account is auto-provisioned from
 * either page with no client-side signal before the redirect
 * (checkFirstLogin.ts is the fallback that later detects this and parks the
 * same `true`). The wording is deliberately phrased for either audience,
 * rather than assuming every reader is mid-registration.
 *
 * Settings (CLO-782) is where that answer can be withdrawn afterward, which
 * is what makes "you can unsubscribe at any time" below actually true.
 *
 * The wording is pinned by MARKETING_CONSENT_SENTENCE and its test: the
 * backend stamps its own version onto every consent record claiming to
 * describe this exact text, so editing it here without bumping
 * MARKETING_CONSENT_TEXT_VERSION in cloud-backend would make every stored
 * record attest to wording nobody saw.
 */
export default function MarketingConsentNotice(): React.ReactElement {
  return (
    <Text size="xs" className="!font-light !leading-[1.5] !text-[#EDECEA]/75 w-full !text-center">
      By continuing, you agree to receive emails from Cognee about product updates, news, and
      events. You can unsubscribe at any time. See our{" "}
      <a
        href={PRIVACY_POLICY_URL}
        target="_blank"
        rel="noopener noreferrer"
        className="!text-[#BC9BFF] underline hover:no-underline"
      >
        Privacy Policy
      </a>
      .
    </Text>
  );
}
