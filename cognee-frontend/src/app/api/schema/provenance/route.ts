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
