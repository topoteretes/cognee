import type { ReactElement } from "react";
import AutoRechargeIllustration from "./AutoRechargeIllustration";

export interface FeatureAnnouncementContent {
  title: string;
  description: string;
  ctaLabel: string;
  ctaHref: string;
  illustration: () => ReactElement;
}

// One entry per feature_key the backend can send in pendingFeatureAnnouncements
// (modules.feature_announcements.feature_launch_dates.FEATURE_LAUNCH_DATES on
// the backend — keys must match exactly). Add a new key here whenever a new
// one is added there; an unrecognized key is simply never shown (see
// FeatureAnnouncementsProvider), so a launch-date entry without matching
// content here fails silently rather than crashing.
export const FEATURE_ANNOUNCEMENT_CONTENT: Record<string, FeatureAnnouncementContent> = {
  auto_recharge_intro: {
    title: "Never run out of credits again",
    description:
      "Turn on auto recharge and we'll top up your workspace automatically whenever your balance runs low — no more surprise 402s mid-upload. Set your own threshold, top-up amount, and a monthly cap, right from Billing.",
    ctaLabel: "Set up auto recharge",
    ctaHref: "/billing",
    illustration: AutoRechargeIllustration,
  },
};
