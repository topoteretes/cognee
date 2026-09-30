"use client";

// Public copy of the SaaS module of the same path. The directory is excluded
// from the sync (it contains cloud-only billing/config actions), but this file
// itself is portable and its consumers are shared UI — keep it compatible with
// the SaaS original; the sync build gate fails if the signatures drift.

import { useCallback, useEffect, useState } from "react";
import {
  getInsufficientCreditsFailure,
  setInsufficientCreditsFailure,
  clearInsufficientCreditsFailure,
  type InsufficientCreditsFailureRecord,
} from "@/utils/browserStorage";

// A failure from days ago isn't actionable context for "your last upload
// just failed" — past this, treat the record as stale and drop it.
const NOTICE_TTL_MS = 24 * 60 * 60 * 1000;

export interface InsufficientCreditsNoticeState {
  isVisible: boolean;
  operation: string | null;
  dismiss: () => void;
  record: (event: { operation: string | null; at: number; tenantId: string | null }) => void;
}

// The one-time mount check surfaces the persistent notice; `record` is what
// the live bridge listener (InsufficientCreditsProvider) calls to persist a
// failure. They're decoupled through localStorage rather than shared React
// state — the notice deliberately doesn't react live to `record`, since a
// failure that just happened is already shown by the reactive modal.
export function useInsufficientCreditsNotice(tenantId: string | null): InsufficientCreditsNoticeState {
  const [notice, setNotice] = useState<InsufficientCreditsFailureRecord | null>(null);

  useEffect(() => {
    if (!tenantId) return;
    const existing = getInsufficientCreditsFailure(tenantId);
    if (!existing) return;
    if (Date.now() - existing.at < NOTICE_TTL_MS) {
      setNotice(existing);
    } else {
      clearInsufficientCreditsFailure(tenantId);
    }
  }, [tenantId]);

  const dismiss = useCallback(() => {
    if (tenantId) clearInsufficientCreditsFailure(tenantId);
    setNotice(null);
  }, [tenantId]);

  // Deliberately writes to event.tenantId, NOT the hook's own `tenantId` —
  // this must be the tenant that actually failed, which may no longer be the
  // active one by the time a delayed 402 arrives (see InsufficientCreditsEvent).
  const record = useCallback(
    (event: { operation: string | null; at: number; tenantId: string | null }) => {
      if (!event.tenantId) return;
      setInsufficientCreditsFailure(event.tenantId, { operation: event.operation, at: event.at });
    },
    [],
  );

  return { isVisible: notice !== null, operation: notice?.operation ?? null, dismiss, record };
}
