import { NextRequest, NextResponse } from "next/server";
import { getServerBackendUrl } from "@/modules/config/serverRuntimeConfig";

export async function GET(request: NextRequest) {
  const localApiUrl = getServerBackendUrl();
  const headers: Record<string, string> = {};
  const cookie = request.headers.get("cookie");
  if (cookie) headers["cookie"] = cookie;
  const authHeader = request.headers.get("authorization");
  if (authHeader) headers["authorization"] = authHeader;
  const apiKey = request.headers.get("x-api-key");
  if (apiKey) headers["x-api-key"] = apiKey;

  // Server-side default-user login, used only when the browser sent no
  // credentials at all. It runs only when DEFAULT_USER_PASSWORD is configured:
  // the default user has no loginable password otherwise, and this route must
  // never carry a publicly-known credential of its own.
  const defaultUserPassword = process.env.DEFAULT_USER_PASSWORD;
  if (!cookie && !authHeader && !apiKey && defaultUserPassword) {
    try {
      const loginResp = await fetch(`${localApiUrl}/api/v1/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          username: process.env.DEFAULT_USER_EMAIL || "default_user@example.com",
          password: defaultUserPassword,
        }).toString(),
      });
      if (loginResp.ok) {
        const data = await loginResp.json();
        headers["authorization"] = `Bearer ${data.access_token}`;
      }
    } catch {
      // Fall through
    }
  }

  try {
    const response = await fetch(`${localApiUrl}/api/v1/schema/provenance`, { headers });

    if (!response.ok) {
      return NextResponse.json({ error: `Backend returned ${response.status}` }, { status: response.status });
    }

    const html = await response.text();
    return new NextResponse(html, {
      headers: { "Content-Type": "text/html" },
    });
  } catch {
    return NextResponse.json({ error: "Failed to reach backend" }, { status: 502 });
  }
}
