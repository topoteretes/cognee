import localFetch from "@/modules/instances/localFetch";

export default async function getMyUserId(): Promise<string | null> {
  try {
    const response = await localFetch("/v1/users/me");
    const user = await response.json();
    return typeof user?.id === "string" ? user.id : null;
  } catch {
    try {
      const response = await localFetch("/v1/auth/me");
      const user = await response.json();
      return typeof user?.id === "string" ? user.id : null;
    } catch {
      return null;
    }
  }
}
