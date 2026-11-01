"use server";

import CogneeUser from "./CogneeUser";

const localApiUrl = process.env.NEXT_PUBLIC_LOCAL_API_URL || "http://localhost:8000";

export default async function getLocalUser(): Promise<CogneeUser | null> {
  // In local mode, we can't use Auth0 session.
  // Return a placeholder user — the actual auth is handled by
  // the backend cookie, not a server-side session.
  // The LocalProvider already verified auth via /api/v1/users/me.
  return {
    id: "local",
    name: "Local User",
    email: "local@cognee.local",
    picture: "",
  };
}
