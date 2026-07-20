"use server";

import type { UserAppState } from "./types";

/**
 * Open-source stub — app_metadata flags require Auth0.
 * Returns defaults so the OSS build treats every user as onboarded.
 */
export default async function getUserAppState(): Promise<UserAppState> {
  return { hasSeen_welcome: true, onboarding_complete: true };
}
