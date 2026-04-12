/**
 * Open-source stub — workspace creation is a cloud-only feature.
 */
export default async function createWorkspace(_tenantName: string): Promise<{ success: boolean; error?: string }> {
  return { success: false, error: "Workspace creation is not available in local mode." };
}
