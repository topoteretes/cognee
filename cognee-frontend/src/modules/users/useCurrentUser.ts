"use client";

import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import getCurrentUserProfile from "./getCurrentUserProfile";
import isCloudEnvironment from "@/utils/isCloudEnvironment";
import type CogneeUser from "./CogneeUser";

export const CURRENT_USER_QUERY_KEY = ["current-user"] as const;

// Shared across TopBar, ProfileWidget, IntercomWidget, and IdentifyUser so the
// Auth0 profile is fetched once per session rather than once per component.
//
// This read powers optional profile UI, so it must never navigate. Middleware
// remains the sole owner of protected-route redirects.
export function useCurrentUser(enabled = true): UseQueryResult<CogneeUser | null> {
  return useQuery({
    queryKey: CURRENT_USER_QUERY_KEY,
    queryFn: async (): Promise<CogneeUser | null> => {
      try {
        return await getCurrentUserProfile();
      } catch {
        return null;
      }
    },
    staleTime: (query) => (query.state.data ? Infinity : 0),
    retry: false,
    throwOnError: false,
    enabled: enabled && isCloudEnvironment(),
  });
}
