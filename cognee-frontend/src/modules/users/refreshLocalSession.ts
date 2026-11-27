import { getLocalApiUrl } from "./getLocalApiUrl";

let pending: Promise<boolean> | null = null;

/**
 * Renew the local backend login with the refresh cookie the login set.
 *
 * The backend's access token lasts an hour; this exchanges the refresh cookie
 * for a new one. Resolves to whether the login was renewed: false means the
 * user has to sign in again. Requests that fail together share one call.
 */
export default function refreshLocalSession(): Promise<boolean> {
  pending ??= global
    .fetch(`${getLocalApiUrl()}/api/v1/auth/refresh`, { method: "POST", credentials: "include" })
    .then((response) => response.ok)
    .finally(() => {
      pending = null;
    });
  return pending;
}
