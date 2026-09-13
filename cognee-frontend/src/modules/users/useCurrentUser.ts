"use client";

import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import getUser from "./getUser";
import isCloudEnvironment from "@/utils/isCloudEnvironment";
import type CogneeUser from "./CogneeUser";

export const CURRENT_USER_QUERY_KEY = ["current-user"] as const;

// Shared across TopBar, ProfileWidget, IntercomWidget, and IdentifyUser so the
// Auth0 profile is fetched once per session rather than once per component.
//
// getUser() is cloud-only and redirects to /sign-in when there's no Auth0
// session. Since it's a Server Action, that redirect fires as a real
// navigation even when the caller wraps it in try/catch — so this must never
// run in local mode, where it would race LocalProvider's own /local-login
// redirect and win.
export function useCurrentUser(enabled = true): UseQueryResult<CogneeUser | null> {
  return useQuery({
    queryKey: CURRENT_USER_QUERY_KEY,
    queryFn: async (): Promise<CogneeUser | null> => {
      try {
        return await getUser();
      } catch {
        // No session (getUser redirects) or the fetch failed — treat as
        // unauthenticated rather than surfacing an error to every consumer.
        return null;
      }
    },
    staleTime: Infinity,
    retry: false,
    throwOnError: false,
    enabled: enabled && isCloudEnvironment(),
  });
}
