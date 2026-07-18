"use client";

import { useState, useEffect } from "react";
import type { CogneeInstance } from "@/modules/instances/types";
import getDatasets from "@/modules/datasets/getDatasets";
import getDatasetData from "@/modules/datasets/getDatasetData";
import { listSessions, SEARCH_SESSION_PREFIX } from "@/modules/sessions/getSessions";

// Detects agent activity while either the Upload or Recall step is active.
// There is NO generic "did we get an API call" endpoint, so we watch two
// concrete signals and flip on whichever appears first:
//   • a NEW session  → recall / session-scoped calls (carry a session_id)
//   • NEW data docs  → graph-direct uploads (no session_id, so no session)
// Both are baselined when the step opens so only activity AFTER that counts.
export function useAgentConnectionDetection(
  cogniInstance: CogneeInstance | null,
  active: boolean,
): boolean {
  const [connectVerified, setConnectVerified] = useState(false);

  useEffect(() => {
    if (!active || !cogniInstance || connectVerified) return;
    let cancelled = false;
    let primed = false;
    const baselineSessions = new Set<string>();
    let baselineDocs = -1;

    const realSessionIds = (rows: { session_id: string }[]) =>
      rows.map((s) => s.session_id).filter((id) => !id.startsWith(SEARCH_SESSION_PREFIX));

    // Doc count of the default dataset (the onboarding target). -1 = unknown.
    async function docCount(): Promise<number> {
      try {
        const datasets = await getDatasets(cogniInstance!);
        if (!Array.isArray(datasets) || datasets.length === 0) return 0;
        const target = datasets.find((d: { name?: string }) => d.name === "default_dataset") ?? datasets[0];
        if (!target?.id) return 0;
        const data = await getDatasetData(target.id, cogniInstance!);
        return Array.isArray(data) ? data.length : 0;
      } catch {
        return -1;
      }
    }

    async function check() {
      const [page, docs] = await Promise.all([
        listSessions(cogniInstance!, { range: "24h", limit: 50 }).catch((err) => {
          console.warn("Onboarding progress check: sessions fetch failed", err);
          return null;
        }),
        docCount(),
      ]);
      if (cancelled) return;
      const ids = page ? realSessionIds(page.sessions) : [];
      if (!primed) {
        ids.forEach((id) => baselineSessions.add(id));
        baselineDocs = docs;
        primed = true;
        return;
      }
      const newSession = ids.some((id) => !baselineSessions.has(id));
      const newDocs = docs >= 0 && baselineDocs >= 0 && docs > baselineDocs;
      if (newSession || newDocs) setConnectVerified(true);
    }

    check();
    const id = setInterval(check, 7000);
    return () => { cancelled = true; clearInterval(id); };
  }, [active, cogniInstance, connectVerified]);

  return connectVerified;
}
