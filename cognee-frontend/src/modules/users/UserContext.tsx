"use client";

import { createContext, useContext } from "react";
import type { SessionRecordingConsentSource } from "@/modules/analytics/sessionRecordingConsent";
import type { ProfileNames } from "./profileName";

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
  // Session-recording consent (A3.1, § 25(1) TDDDG). Two fields because
  // "never asked" and "declined" both mean do not record, but only the first
  // means ask: ConsentedSessionReplay gates the replay tools on
  // sessionRecordingOptIn, and the consent dialog appears on
  // sessionRecordingConsentAnswered being false.
  sessionRecordingOptIn: boolean;
  // Three states, not two. null is "the backend did not tell us", which is
  // what a deployment without the consent endpoint looks like, and it must
  // not read as "this user has never answered" — that renders a modal whose
  // only two buttons both write to an endpoint that is not there, leaving a
  // dialog no one can get past. Only an explicit false asks.
  sessionRecordingConsentAnswered: boolean | null;
  // The user's own name (CLO-456). Three states, like the consent answer above:
  // null is "the backend did not tell us" (a backend that predates the field, or
  // /me that failed), and a profile whose names are both null is an account that
  // has not set a name yet, which is every existing account. The difference
  // decides what the profile form does: null makes it read-only (a save would go
  // to an endpoint that may not be there), while two nulls leave it editable with
  // empty inputs.
  profile: ProfileNames | null;
}

// A workspace the current user belongs to. Owned by the User domain (it answers
// "what am I a member of"), not the Tenant domain (which owns the ACTIVE
// tenant's operational state — apiKey, pod health, cogniInstance).
export interface AvailableTenant {
  id: string;
  name: string;
  isOwner: boolean;
  ownerHasSubscription: boolean;
  // Control-plane lifecycle status, passed straight through from
  // /api/my-tenants. Typed loosely on purpose: the tenant module owns the
  // precise union (TenantLifecycleStatus in modules/tenant/types.ts) and this
  // module never imports from it, matching the one-way users <- tenant
  // dependency the rest of the codebase follows. Consumers compare against
  // TENANT_STATUS_SUSPENDED; absent means "assume active".
  status?: string;
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
  // Records this user's answer about session recording and updates the cache
  // with what the server stored, never with what was asked for: a write the
  // backend's write-once guard refuses comes back as the stored answer, and
  // the gate must obey that rather than the request.
  setSessionRecordingConsent: (
    optedIn: boolean,
    source: SessionRecordingConsentSource,
  ) => Promise<void>;
  // Saves the name and updates the cache with what the server stored, never
  // with what was typed: the backend normalizes it, so the stored form is the
  // one the rest of the app must show. Only the fields passed are changed
  // (null clears one). Always sent as a settings write: the profile is the one
  // place a name is edited. Rejects with an HttpError, status 422 when the
  // backend refuses the characters.
  updateProfileName: (names: Partial<ProfileNames>) => Promise<void>;
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
  setSessionRecordingConsent: async () => {},
  updateProfileName: async () => {},
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
