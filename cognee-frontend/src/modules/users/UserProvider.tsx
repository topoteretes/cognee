"use client";

import { useEffect, useCallback, type ReactNode } from "react";
import { usePathname } from "next/navigation";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { http, HttpError } from "@/services/http";
import { isAuthRoutePath } from "@/utils/sessionRoutes";
import { ensureSessionFingerprint } from "@/utils/sessionScopedStorage";
import { subscribeToSessionRecordingWithdrawal } from "@/utils/sessionRecordingWithdrawal";
import { UserContext, type UserMe, type AvailableTenant } from "./UserContext";
import type { SessionRecordingConsentSource } from "@/modules/analytics/sessionRecordingConsent";
import type { ProfileNames } from "./profileName";

interface Props {
  children: ReactNode;
}

export const ME_QUERY_KEY = ["me"] as const;
export const TENANTS_QUERY_KEY = ["my-tenants"] as const;

export default function UserProvider({ children }: Props): ReactNode {
  const pathname = usePathname();
  const queryClient = useQueryClient();

  // No session exists yet on auth pages — calling /api/me there is pointless
  // (it just 401s against the management API) and it's the request the user
  // spotted firing from /sign-in.
  const isAuthPath = isAuthRoutePath(pathname);

  const { data: userMe = null, isLoading, isError: isUserMeError } = useQuery({
    queryKey: ME_QUERY_KEY,
    queryFn: () => http.getJson<UserMe>("/api/me"),
    staleTime: Infinity,
    retry: false,
    throwOnError: false,
    enabled: !isAuthPath,
  });

  // Which workspaces this user belongs to — a User-domain fact ("what am I a
  // member of"), fetched here alongside /api/me rather than in TenantProvider
  // (which only owns the ACTIVE tenant's operational state).
  const { data: availableTenants = [], isLoading: isLoadingTenants, isError: isTenantsError } = useQuery({
    queryKey: TENANTS_QUERY_KEY,
    queryFn: () => http.getJson<AvailableTenant[]>("/api/my-tenants"),
    staleTime: 60_000,
    retry: false,
    throwOnError: false,
    enabled: !isAuthPath,
  });

  // Entering an auth page means the session this tab was using is over (it
  // expired, or the user signed out). Signing in again as someone else is a soft
  // navigation, so this provider stays mounted, and with staleTime Infinity the
  // previous account's /me would be served as current: the session fingerprint
  // below would keep comparing against the OLD user and never wipe their cached
  // workspace, API key and flags. Forgetting both answers here makes the first
  // non-auth page ask again, and the fingerprint then sees the new account.
  useEffect(() => {
    if (!isAuthPath) return;
    void queryClient.resetQueries({ queryKey: ME_QUERY_KEY });
    void queryClient.resetQueries({ queryKey: TENANTS_QUERY_KEY });
  }, [isAuthPath, queryClient]);

  // General backstop against cross-account storage leakage: sessionStorage
  // and localStorage are scoped to the browser, not the account, so anything
  // they cache (selected tenant, init cache, plan type, awaiting-dataset
  // flags, ...) belongs to whoever was logged in when it was written. A
  // second account logging in on the same tab — or a stale tab that missed
  // the signout script — would otherwise inherit it. This runs before
  // anything else has a chance to read that stale state.
  useEffect(() => {
    ensureSessionFingerprint(userMe?.userId);
  }, [userMe]);

  // Another tab withdrew session-recording consent. Correcting the cache here
  // rather than in the gate keeps the decision in the one place that owns it:
  // ConsentedSessionReplay then stops Sentry's replay and unmounts the
  // Contentsquare reporting on the next render, the same path a withdrawal in
  // this tab takes.
  //
  // This tab is NOT reloaded. Only a reload removes the Contentsquare tag
  // itself, and reloading a tab the user is not looking at would discard
  // whatever they had typed in it to remove a tag that stops being fed
  // anything the moment this runs. That residue is the documented cost, and
  // it ends at that tab's next navigation.
  useEffect(() => {
    const userId = userMe?.userId ?? null;
    if (userId === null) return;
    return subscribeToSessionRecordingWithdrawal(userId, () => {
      queryClient.setQueryData<UserMe>(ME_QUERY_KEY, (current) =>
        current ? { ...current, sessionRecordingOptIn: false, sessionRecordingConsentAnswered: true } : current,
      );
    });
  }, [userMe?.userId, queryClient]);

  const markWelcomeSeen = useCallback(async (): Promise<void> => {
    const updated = await http.patchJson<UserMe>("/api/me/seen-welcome");
    queryClient.setQueryData<UserMe>(ME_QUERY_KEY, updated);
  }, [queryClient]);

  // Onboarding is shown once per user, ever — not scoped to a workspace, so
  // creating or switching workspaces never re-triggers it. This cache only
  // feeds UI: the welcome/onboarding redirects are decided on the server
  // ((app)/layout.tsx, dashboard/page.tsx, welcome/page.tsx and
  // onboarding/page.tsx), never here.
  const markOnboardingComplete = useCallback(async (): Promise<void> => {
    const updated = await http.patchJson<UserMe>("/api/me/complete-onboarding");
    queryClient.setQueryData<UserMe>(ME_QUERY_KEY, updated);
  }, [queryClient]);

  // Sends the source the caller states rather than deriving one, because the
  // two are not interchangeable on the backend: "first_login" is write-once
  // and "settings" always wins. The response is the stored answer, so a
  // refused duplicate corrects the cache instead of confirming the request.
  const setSessionRecordingConsent = useCallback(
    async (optedIn: boolean, source: SessionRecordingConsentSource): Promise<void> => {
      try {
        const updated = await http.patchJson<UserMe>("/api/me/session-recording-consent", {
          optedIn,
          source,
        });
        queryClient.setQueryData<UserMe>(ME_QUERY_KEY, updated);
      } catch (e) {
        // A 404 is the one failure that is not a failed save: the route
        // returns it only for consent_endpoint_not_deployed, so there is no
        // answer to store and no question worth asking. Moving the cache to
        // the unknown state closes the dialog for this session instead of
        // leaving the user retrying a write that cannot succeed. Every other
        // status still throws, so a genuinely failed save keeps the dialog
        // open rather than being silently treated as answered.
        if (e instanceof HttpError && e.status === 404) {
          queryClient.setQueryData<UserMe>(ME_QUERY_KEY, (current) =>
            current ? { ...current, sessionRecordingConsentAnswered: null } : current,
          );
          return;
        }
        throw e;
      }
    },
    [queryClient],
  );

  // Writes only the fields it is given. The response is the stored profile, so
  // the cache shows the backend's normalized form of the name, not the typed one.
  const updateProfileName = useCallback(
    async (names: Partial<ProfileNames>): Promise<void> => {
      const updated = await http.patchJson<UserMe>("/api/me/profile", { ...names, source: "settings" });
      queryClient.setQueryData<UserMe>(ME_QUERY_KEY, updated);
    },
    [queryClient],
  );

  const dismissFeatureAnnouncement = useCallback(async (featureKey: string): Promise<void> => {
    const updated = await http.patchJson<UserMe>(`/api/me/feature-announcements/${featureKey}/dismiss`);
    queryClient.setQueryData<UserMe>(ME_QUERY_KEY, updated);
  }, [queryClient]);

  const refetchTenants = useCallback((): void => {
    queryClient.invalidateQueries({ queryKey: TENANTS_QUERY_KEY });
  }, [queryClient]);

  const setAvailableTenantsOptimistic = useCallback((tenants: AvailableTenant[]): void => {
    queryClient.setQueryData<AvailableTenant[]>(TENANTS_QUERY_KEY, tenants);
  }, [queryClient]);

  return (
    <UserContext.Provider
      value={{
        userMe,
        isLoading,
        isUserMeError,
        markWelcomeSeen,
        markOnboardingComplete,
        dismissFeatureAnnouncement,
        setSessionRecordingConsent,
        updateProfileName,
        availableTenants,
        isLoadingTenants,
        isTenantsError,
        refetchTenants,
        setAvailableTenantsOptimistic,
      }}
    >
      {children}
    </UserContext.Provider>
  );
}
