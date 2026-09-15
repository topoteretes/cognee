"use client";

import { useTenant } from "@/modules/tenant/TenantProvider";
import { trackEvent } from "@/modules/analytics";

type TrackEventParams = Parameters<typeof trackEvent>[0];

// Wraps trackEvent to stamp every onboarding event with the active tenant_id.
// Read directly off TenantContext (not sessionStorage, unlike plan_type/
// user_email) so it always reflects the workspace actually being onboarded,
// even if it changes mid-flow — a plain cache would risk going stale given
// this codebase's known tenant-context race conditions.
export function useOnboardingTrackEvent(): (params: TrackEventParams) => void {
  const { tenant } = useTenant();
  const tenantId = tenant?.tenant_id;

  return function track(params: TrackEventParams): void {
    trackEvent({
      ...params,
      additionalProperties: {
        ...(tenantId ? { tenant_id: tenantId } : {}),
        ...params.additionalProperties,
      },
    });
  };
}
