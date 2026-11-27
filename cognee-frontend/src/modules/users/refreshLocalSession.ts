import { toHttpError } from "@/services/http/errors";

import { getLocalApiUrl } from "./getLocalApiUrl";

let pending: Promise<boolean> | null = null;

/**
 * Renew the local backend login with the refresh cookie the login set.
 *
 * The backend's access token lasts an hour; this exchanges the refresh cookie
 * for a new one. Resolves to whether the login was renewed: false means the
 * backend refused the cookie and the user has to sign in again. Any other
 * failure says nothing about the login and is thrown, so a backend error is
 * shown as one instead of sending the user to the login page. Requests that
 * fail together share one call.
 */
export default function refreshLocalSession(): Promise<boolean> {
  pending ??= global
    .fetch(`${getLocalApiUrl()}/api/v1/auth/refresh`, { method: "POST", credentials: "include" })
    .then((response) => {
      if (response.ok) return true;
      if (response.status === 401) return false;
      return toHttpError(response);
    })
    .finally(() => {
      pending = null;
    });
  return pending;
}
