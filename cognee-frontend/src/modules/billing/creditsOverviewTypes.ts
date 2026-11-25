// Types for the public copy of getCreditsOverview.
//
// They live in their own plain module rather than in getCreditsOverview.ts
// because that file is "use server": Next's server-actions loader analyses its
// exports syntactically, so a type exported from there can end up emitted as a
// runtime binding and throw ReferenceError on module evaluation — taking down
// every server-side importer. A "use server" module exports async functions and
// nothing else.
//
// The SaaS original keeps the same shapes in its payment infrastructure tree,
// which this copy cannot import: that tree is excluded from the sync and its
// import alias is on the blocked list in sync-to-public.sh. Keep the two in
// step by hand.
//
// The alias is spelled out nowhere in this file on purpose. Phase 9 of the sync
// greps the synced tree for the literal string with `grep -F`, with no notion of
// whether it sits in an import or in prose, so naming it here fails the very
// check it is describing.

export interface CreditsOverview {
  budget: { totalUsd: number; spentUsd: number | null; remainingUsd: number | null; spentPct: number | null };
  tenants: { tenantId: string; tenantName: string; spentUsd: number | null; maxBudgetUsd: number | null; remainingUsd: number | null; purchasedUsd: number | null; initialBudgetUsd: number | null }[];
  purchases: { id: string; tokensPurchased: number; amountPaidUsd: number; currency: string; createdAt: string; tenantId: string | null }[];
}

// This build only ever answers `unavailable`: an open-source build has no
// billing backend, which is a different fact from a request that failed.
export type CreditsOverviewResult =
  | { status: "ok"; data: CreditsOverview }
  | { status: "error"; message: string }
  | { status: "unavailable" };
