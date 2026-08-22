"use client";

import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import getUser from "./getUser";
import type CogneeUser from "./CogneeUser";

export const CURRENT_USER_QUERY_KEY = ["current-user"] as const;

// Shared across TopBar, ProfileWidget, IntercomWidget, and IdentifyUser so the
// Auth0 profile is fetched once per session rather than once per component.
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
    enabled,
  });
}
