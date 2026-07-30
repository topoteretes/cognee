import { NextResponse } from "next/server";
import getTenantProvisioningStatus from "@/modules/tenant/getTenantProvisioningStatus";

// Route handler (not a server action) so waitForPodReady's tight polling loop
// doesn't serialize against other concurrent server action calls (same
// reasoning as /api/create-workspace and /api/my-tenants) — and so each call
// shows up as its own named request in the network tab instead of being
// folded into an opaque POST to the current page.
export async function GET(request: Request) {
  const tenantId = new URL(request.url).searchParams.get("tenantId");
  if (!tenantId) {
    return NextResponse.json({ error: "tenantId is required" }, { status: 400 });
  }

  const status = await getTenantProvisioningStatus(tenantId);
  return NextResponse.json({ status });
}
