"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import { useUser } from "@/modules/users/UserContext";

/**
 * Side-effect-only hook. Onboarding is shown exactly once per user, ever —
 * right after welcome, never re-forced based on activity, and never reset
 * by creating or switching workspaces. onboardingCompletedAt (from /me) is
 * the only signal; there is no local fallback, so it survives sign-out and
 * follows the user across devices.
 */
export function useOnboardingRedirect(): void {
  const { isInitializing } = useCogniInstance();
  const { tenant } = useTenant();
  const { userMe } = useUser();
  const router = useRouter();

  useEffect(() => {
    // tenant === null means a new user is in the provisioning flow —
    // TenantProvider redirects to /welcome; don't race it here.
    if (isInitializing || !tenant) return;
    // userMe === null means /me hasn't resolved yet — an unknown, not an
    // answer; redirecting on it would send an already-onboarded user into
    // onboarding on every transient fetch delay.
    if (userMe === null || userMe.onboardingCompletedAt) return;
    router.replace("/onboarding");
  }, [isInitializing, tenant, userMe, router]);
}
