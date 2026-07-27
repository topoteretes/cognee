"use client";

import { useEffect, useCallback, type ReactNode } from "react";
import { useRouter, usePathname } from "next/navigation";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { http } from "@/services/http";
import { isAuthRoutePath } from "@/utils/sessionRoutes";
import { ensureSessionFingerprint } from "@/utils/sessionScopedStorage";
import { UserContext, type UserMe, type AvailableTenant } from "./UserContext";

interface Props {
  children: ReactNode;
}

const SETUP_PATHS = ["/welcome", "/onboarding"];

export const ME_QUERY_KEY = ["me"] as const;
export const TENANTS_QUERY_KEY = ["my-tenants"] as const;

export default function UserProvider({ children }: Props): ReactNode {
  const router = useRouter();
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

  useEffect(() => {
    if (userMe === null || userMe.isSeenWelcome) return;
    if (SETUP_PATHS.some((p) => pathname.startsWith(p))) return;
    router.replace("/welcome");
    // No reactive "bounce off /welcome once seen" branch here on purpose:
    // markWelcomeSeen's optimistic cache update can flip isSeenWelcome to
    // true while pathname is still "/welcome" (the router.push to
    // /onboarding hasn't committed a pathname change yet), which raced this
    // effect into redirecting to /dashboard instead — a promise-ordering
    // race, not an occasional timing fluke, so it hit reliably. The
    // server-side check in welcome/page.tsx already redirects an
    // already-onboarded user away from /welcome on any real page load;
    // that's the only case this branch was for.
  }, [userMe, pathname, router]);

  const markWelcomeSeen = useCallback(async (): Promise<void> => {
    const updated = await http.patchJson<UserMe>("/api/me/seen-welcome");
    queryClient.setQueryData<UserMe>(ME_QUERY_KEY, updated);
  }, [queryClient]);

  // Onboarding is shown once per user, ever — not scoped to a workspace, so
  // creating or switching workspaces never re-triggers it. The redirect
  // decision (useOnboardingRedirect) reads onboardingCompletedAt directly
  // off this cache; this is the only place that sets it.
  const markOnboardingComplete = useCallback(async (): Promise<void> => {
    const updated = await http.patchJson<UserMe>("/api/me/complete-onboarding");
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
