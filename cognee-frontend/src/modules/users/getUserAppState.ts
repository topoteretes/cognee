"use server";

/** Auth0 app_metadata flags. Always defaulted in the open-source build. */
export interface UserAppState {
  hasSeen_welcome: boolean;
  onboarding_complete: boolean;
}

/**
 * Open-source stub — app_metadata flags require Auth0.
 * Returns defaults so the OSS build treats every user as onboarded.
 */
export default async function getUserAppState(): Promise<UserAppState> {
  return { hasSeen_welcome: true, onboarding_complete: true };
}
