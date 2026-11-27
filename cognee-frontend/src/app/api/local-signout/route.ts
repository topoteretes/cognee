import { NextResponse } from "next/server";

import { getServerBackendUrl } from "@/modules/config/serverRuntimeConfig";

export async function GET(request: Request) {
  const localApiUrl = getServerBackendUrl();
  const response = NextResponse.redirect(new URL("/local-login", request.url));

  // The backend ends the login and answers with the deletion of its cookies
  // (the access and the refresh cookie). Pass those on to the browser: only
  // the backend knows the cookies' names and attributes.
  let logoutResponse: Response;
  try {
    logoutResponse = await fetch(`${localApiUrl}/api/v1/auth/logout`, {
      method: "POST",
      headers: {
        cookie: request.headers.get("cookie") || "",
      },
    });
  } catch {
    // Backend might be down — still redirect
    return response;
  }

  for (const cookie of logoutResponse.headers.getSetCookie()) {
    response.headers.append("set-cookie", cookie);
  }
  return response;
}
