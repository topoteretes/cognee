"use client";

import { useQuery } from "@tanstack/react-query";
import { http } from "@/services/http";
import { parseReleaseLog } from "./releaseLog";
import type { ReleaseLog } from "./releaseLog";

export const RELEASES_QUERY_KEY = ["releases"] as const;

/** The release log, shared by the help menu and the /releases page. The backend
 *  caches it for 15 minutes; the client holds it for as long, so opening the
 *  help menu costs nothing after the first time. Never throws: a failure
 *  leaves `data` undefined and the UI hides the section. */
export function useReleases() {
  return useQuery({
    queryKey: RELEASES_QUERY_KEY,
    queryFn: () => http.getJson<ReleaseLog>("/api/releases", { validate: parseReleaseLog }),
    staleTime: 15 * 60_000,
    retry: false,
    throwOnError: false,
  });
}
