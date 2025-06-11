import { redirect } from "next/navigation";
import { auth0 } from "@/modules/auth/auth0";

export async function GET(request: Request) {
  const accessToken = await auth0.getAccessToken();

  if (accessToken) {
    const response = new Response();

    response.headers.set("Set-Cookie", `${process.env.AUTH_TOKEN_COOKIE_NAME}=${accessToken.token}; Expires=${new Date(accessToken.expiresAt * 1000).toUTCString()}; Path=/; SameSite=Lax; Domain=localhost; HttpOnly`);

    return response;
  } else {
    redirect("/auth");
  }
}
