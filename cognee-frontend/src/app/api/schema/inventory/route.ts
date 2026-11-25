import { NextRequest, NextResponse } from "next/server";

import { getServerBackendUrl } from "@/modules/config/serverRuntimeConfig";

export async function GET(request: NextRequest) {
  const localApiUrl = getServerBackendUrl();
  const { searchParams } = request.nextUrl;
  const datasetId = searchParams.get("dataset_id");
  console.log("[api/schema/inventory] request received, dataset_id:", datasetId);

  if (!datasetId) {
    return NextResponse.json({ error: "dataset_id required" }, { status: 400 });
  }

  const headers: Record<string, string> = {};
  const cookie = request.headers.get("cookie");
  if (cookie) headers["cookie"] = cookie;
  const authHeader = request.headers.get("authorization");
  if (authHeader) headers["authorization"] = authHeader;
  const apiKey = request.headers.get("x-api-key");
  if (apiKey) headers["x-api-key"] = apiKey;

  const samplesPerType = searchParams.get("samples_per_type") ?? "5";
  const sort = searchParams.get("sort") ?? "count";
  const backendUrl = `${localApiUrl}/api/v1/schema/inventory?dataset_id=${datasetId}&samples_per_type=${samplesPerType}&sort=${sort}`;
  console.log("[api/schema/inventory] forwarding to:", backendUrl);

  try {
    const response = await fetch(backendUrl, { headers });
    console.log("[api/schema/inventory] backend response status:", response.status);

    if (!response.ok) {
      const body = await response.text().catch(() => "(unreadable)");
      console.error("[api/schema/inventory] backend error body:", body);
      return NextResponse.json({ error: `Backend returned ${response.status}`, detail: body }, { status: response.status });
    }

    const data = await response.json();
    console.log("[api/schema/inventory] success, items:", Array.isArray(data) ? data.length : typeof data);
    return NextResponse.json(data);
  } catch (err) {
    console.error("[api/schema/inventory] fetch threw:", err);
    return NextResponse.json({ error: "Failed to reach backend", detail: String(err) }, { status: 502 });
  }
}
