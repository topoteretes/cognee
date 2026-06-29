"use server";

import CogneeUser from "./CogneeUser";
import { cookies } from "next/headers";

const localApiUrl = process.env.NEXT_PUBLIC_LOCAL_API_URL || "http://localhost:8000";

export default async function getLocalUser(): Promise<CogneeUser | null> {
  try {
    const cookieStore = await cookies();
    const authToken = cookieStore.get("auth_token")?.value;
    if (!authToken) return null;

    const response = await fetch(`${localApiUrl}/api/v1/users/me`, {
      headers: { Cookie: `auth_token=${authToken}` },
    });

    if (!response.ok) return null;

    const user = await response.json();
    return {
      id: user.id,
      name: user.email.split("@")[0],
      email: user.email,
      picture: "",
    };
  } catch {
    return null;
  }
}
