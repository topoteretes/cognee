"use server";

import type CogneeUser from "./CogneeUser";
import getLocalUser from "./getLocalUser";

/**
 * Open-source version delegates to the local profile because Auth0 is cloud-only.
 */
export default async function getCurrentUserProfile(): Promise<CogneeUser | null> {
  return getLocalUser();
}
