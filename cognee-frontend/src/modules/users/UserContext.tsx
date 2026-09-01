"use client";

import { createContext, useContext } from "react";

export interface UserMe {
  isSeenWelcome: boolean;
  // null means never completed. Onboarding is a once-per-user fact — not
  // scoped to a workspace, so switching or creating a workspace never
  // affects it.
  onboardingCompletedAt: string | null;
  // Mirrors the backend's own notion of "current tenant" (user.tenant_id) —
  // deliberately not a second, client-owned source of truth.
  activeWorkspaceId: string | null;
  // Auth0's `sub` — a session fingerprint, not an app-domain fact. Lets
  // UserProvider detect a different account logging in on the same tab and
  // wipe the previous account's cached tenant/workspace state.
  userId: string | null;
  // Feature announcement keys this user hasn't dismissed yet (CLO-358).
  // Server-computed — already filtered to only features launched after this
  // user's account was created, so a brand-new signup never sees one.
  pendingFeatureAnnouncements: string[];
  // ISO timestamp, inherited from the backend's Principal row (CLO-363) —
  // lets time-based triggers (e.g. the NPS survey's "15 days old" condition)
  // gate off account age without a separate call. Null for the rare
  // legacy/backfilled row whose Principal.created_at was never set.
  accountCreatedAt: string | null;
}

// A workspace the current user belongs to. Owned by the User domain (it answers
// "what am I a member of"), not the Tenant domain (which owns the ACTIVE
// tenant's operational state — apiKey, pod health, cogniInstance).
export interface AvailableTenant {
  id: string;
  name: string;
  isOwner: boolean;
  ownerHasSubscription: boolean;
}

interface UserContextValue {
  userMe: UserMe | null;
  isLoading: boolean;
  // True when the /me fetch itself failed (network error, non-2xx) rather
  // than genuinely still loading. Without this, retry:false + a permanently
  // null userMe left consumers gated on "userMe === null" stuck behind an
  // infinite spinner with no error and no way to retry.
  isUserMeError: boolean;
  markWelcomeSeen: () => Promise<void>;
  markOnboardingComplete: () => Promise<void>;
  dismissFeatureAnnouncement: (featureKey: string) => Promise<void>;
  availableTenants: AvailableTenant[];
  isLoadingTenants: boolean;
  // True when the tenants fetch itself failed (network error, non-2xx) rather
  // than genuinely returning zero tenants — callers must NOT treat this the
  // same as "user has 0 workspaces", or a transient fetch failure looks
  // identical to a brand-new signup and triggers auto-provisioning for an
  // established user.
  isTenantsError: boolean;
  refetchTenants: () => void;
  // Writes straight into the tenants query's cache — for instant UI feedback
  // right after a mutation (create/switch), ahead of the background refetch
  // that reconciles with the server.
  setAvailableTenantsOptimistic: (tenants: AvailableTenant[]) => void;
}

const DEFAULT: UserContextValue = {
  userMe: null,
  isLoading: true,
  isUserMeError: false,
  markWelcomeSeen: async () => {},
  markOnboardingComplete: async () => {},
  dismissFeatureAnnouncement: async () => {},
  availableTenants: [],
  isLoadingTenants: true,
  isTenantsError: false,
  refetchTenants: () => {},
  setAvailableTenantsOptimistic: () => {},
};

export const UserContext = createContext<UserContextValue>(DEFAULT);

export function useUser(): UserContextValue {
  return useContext(UserContext);
}
