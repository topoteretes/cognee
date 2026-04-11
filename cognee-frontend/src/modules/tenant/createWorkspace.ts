import localFetch from "@/modules/instances/localFetch";

export default async function createWorkspace(tenantName: string): Promise<{ success: boolean; error?: string }> {
  try {
    const response = await localFetch(`/tenants?tenant_name=${encodeURIComponent(tenantName)}`, {
      method: "POST",
    });

    if (!response.ok) {
      const text = await response.text();
      let message = "Failed to create workspace";
      try {
        const json = JSON.parse(text);
        message = json.detail || json.error || json.message || message;
      } catch {
        if (text) message = text;
      }
      return { success: false, error: message };
    }

    return { success: true };
  } catch (err) {
    const message = err instanceof Error ? err.message : "Failed to create workspace";
    return { success: false, error: message };
  }
}
