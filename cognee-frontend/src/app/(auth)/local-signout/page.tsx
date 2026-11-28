"use client";

import { useEffect } from "react";

import { getLocalApiUrl } from "@/modules/users/getLocalApiUrl";
import { navigateTo } from "@/utils/navigateTo";

/**
 * Sign out of the local backend, then go to the login page.
 *
 * The request goes from the browser to the backend: the refresh cookie is sent
 * to the backend's auth routes only, and the backend answers with the deletion
 * of its cookies, which only it knows the names and attributes of.
 */
export default function LocalSignOutPage() {
  useEffect(() => {
    global
      .fetch(`${getLocalApiUrl()}/api/v1/auth/logout`, { method: "POST", credentials: "include" })
      // Backend might be down — still leave the app.
      .catch(() => {})
      .finally(() => navigateTo("/local-login"));
  }, []);

  return null;
}
