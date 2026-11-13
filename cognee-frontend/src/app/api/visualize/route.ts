import { NextRequest, NextResponse } from "next/server";

import { getServerBackendUrl } from "@/modules/config/serverRuntimeConfig";

export async function GET(request: NextRequest) {
  const localApiUrl = getServerBackendUrl();
  const datasetId = request.nextUrl.searchParams.get("dataset_id");
  if (!datasetId) {
    return NextResponse.json({ error: "dataset_id required" }, { status: 400 });
  }

  // Forward all auth headers/cookies the browser sent
  const headers: Record<string, string> = {};
  const cookie = request.headers.get("cookie");
  if (cookie) headers["cookie"] = cookie;
  const authHeader = request.headers.get("authorization");
  if (authHeader) headers["authorization"] = authHeader;
  const apiKey = request.headers.get("x-api-key");
  if (apiKey) headers["x-api-key"] = apiKey;

  try {
    const response = await fetch(
      `${localApiUrl}/api/v1/visualize?dataset_id=${datasetId}`,
      { headers }
    );

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
