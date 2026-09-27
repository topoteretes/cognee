import { NextRequest, NextResponse } from "next/server";
import { getServerBackendUrl } from "@/modules/config/serverRuntimeConfig";

// Proxies the cognee /v1/schema/provenance HTML view (memory-provenance graph:
// Tenant -> User -> Agent -> Brain -> File -> memory) so it can be embedded in
// an iframe. Mirrors the /api/visualize proxy, including default-user login.
export async function GET(request: NextRequest) {
  const localApiUrl = getServerBackendUrl();
  const headers: Record<string, string> = {};
  const cookie = request.headers.get("cookie");
  if (cookie) headers["cookie"] = cookie;
  const authHeader = request.headers.get("authorization");
  if (authHeader) headers["authorization"] = authHeader;
  const apiKey = request.headers.get("x-api-key");
  if (apiKey) headers["x-api-key"] = apiKey;

  // If no auth available from headers, try to login as default user server-side
  // Server-side default-user login, used only when the browser sent no
  // credentials at all. DEFAULT_USER_PASSWORD is the configured password;
  // the literal is the local dev-stack value (`cognee-cli -ui`, docker-compose.yml),
  // kept so deployments created before SDK-549 keep working. Against a server
  // whose default user has no password this attempt simply fails and the
  // request is forwarded unauthenticated, exactly as it would with no fallback.
  const defaultUserPassword = process.env.DEFAULT_USER_PASSWORD || "default_password";
  if (!cookie && !authHeader && !apiKey) {
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
    return new NextResponse(html, { headers: { "Content-Type": "text/html" } });
  } catch {
    return NextResponse.json({ error: "Failed to reach backend" }, { status: 502 });
  }
}
