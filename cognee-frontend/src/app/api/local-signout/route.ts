import { uiPath } from "@/modules/config/uiBasePath";
import { NextResponse } from "next/server";

import { getServerBackendUrl } from "@/modules/config/serverRuntimeConfig";

export async function GET(request: Request) {
  const localApiUrl = getServerBackendUrl();
  // Call the local backend's logout endpoint to invalidate the session
  try {
    await fetch(`${localApiUrl}/api/v1/auth/logout`, {
      method: "POST",
      headers: {
        cookie: request.headers.get("cookie") || "",
      },
    });
  } catch {
    // Backend might be down — still clear cookies and redirect
  }

  // Clear the fastapiusersauth cookie and redirect to local login
  const response = new NextResponse(null, {
    status: 307,
    headers: { Location: uiPath("/local-login") },
  });
  response.cookies.set("fastapiusersauth", "", {
    maxAge: 0,
    path: "/",
  });
  return response;
}
