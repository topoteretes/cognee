"use client";

import { useCallback, type ReactNode } from "react";
import { useUser } from "@/modules/users/UserContext";
import { FEATURE_ANNOUNCEMENT_CONTENT } from "@/modules/featureAnnouncements/featureAnnouncementContent";
import FeatureAnnouncementModal from "@/ui/elements/FeatureAnnouncementModal";

// Single mount point (in the (app) layout, alongside InsufficientCreditsProvider)
// that shows at most one pending feature announcement at a time — the first
// key in userMe.pendingFeatureAnnouncements that we have content for. Keys
// the frontend doesn't recognize (a launch-date entry added on the backend
// before its content lands here) are skipped rather than shown blank.
export default function FeatureAnnouncementsProvider({ children }: { children: ReactNode }): React.ReactElement {
  const { userMe, dismissFeatureAnnouncement } = useUser();

  // Not through welcome yet means a brand-new account on its way to /welcome,
  // which is no moment for an announcement modal.
  const featureKey = userMe?.isSeenWelcome
    ? userMe.pendingFeatureAnnouncements.find((key) => key in FEATURE_ANNOUNCEMENT_CONTENT)
    : undefined;

  const handleDismiss = useCallback((): void => {
    if (featureKey) void dismissFeatureAnnouncement(featureKey);
  }, [featureKey, dismissFeatureAnnouncement]);

  return (
    <>
      {children}
      {featureKey && (
        <FeatureAnnouncementModal
          content={FEATURE_ANNOUNCEMENT_CONTENT[featureKey]}
          onDismiss={handleDismiss}
        />
      )}
    </>
  );
}
