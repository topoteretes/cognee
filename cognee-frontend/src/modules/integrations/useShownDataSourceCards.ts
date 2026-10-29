"use client";

import { useEffect, useState } from "react";
import { DATA_SOURCE_CARDS, isProviderShownOnHost } from "./dataSourceCards";
import type { TeamConnectorCfg } from "./types";

/**
 * The cards this host lists. The host is only known after mount, so SSR and
 * first paint list the released cards (no hydration mismatch), and a dev-only
 * card only ever appears — it never flashes on staging or production.
 */
export function useShownDataSourceCards(): TeamConnectorCfg[] {
  const [host, setHost] = useState<string | null>(null);

  useEffect(() => {
    setHost(window.location.hostname);
  }, []);

  return DATA_SOURCE_CARDS.filter((cfg) => isProviderShownOnHost(cfg.key, host));
}
