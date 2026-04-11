import { NextRequest, NextResponse } from "next/server";
import managementFetch from "@/modules/instances/managementFetch";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const { tenantId } = await params;
    const response = await managementFetch(
      `/permissions/tenants/${tenantId}/users`,
      { method: "GET" },
      { noRedirectOnAuth: true }
    );
    const data = await response.json();
    return NextResponse.json(data);
  } catch {
    // Return empty array when user doesn't have management access (e.g. guest on another tenant)
    return NextResponse.json([]);
  }
}
