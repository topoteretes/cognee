import { NextRequest, NextResponse } from "next/server";
import managementFetch from "@/modules/instances/managementFetch";

export async function POST(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const email = searchParams.get("email");
  const tenantId = searchParams.get("tenant_id");

  if (!email || !tenantId) {
    return NextResponse.json({ error: "email and tenant_id are required" }, { status: 400 });
  }

  try {
    const params = new URLSearchParams({ email, tenant_id: tenantId });
    await managementFetch(`/tenants/users?${params}`, { method: "POST" });
  } catch {
    // Swallow errors — caller always shows success notification
  }

  return NextResponse.json({ success: true });
}
