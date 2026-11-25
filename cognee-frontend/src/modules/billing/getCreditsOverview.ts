"use server";

import type { CreditsOverviewResult } from "./creditsOverviewTypes";

/**
 * Open-source stub — credit overview requires the cloud billing backend.
 * Reports `unavailable` so consumers skip the balance UI entirely, without
 * ever presenting it to the user as an error.
 *
 * Nothing but the async function is exported from here — see
 * creditsOverviewTypes.ts for why the shapes live elsewhere.
 */
export default async function getCreditsOverview(): Promise<CreditsOverviewResult> {
  return { status: "unavailable" };
}
