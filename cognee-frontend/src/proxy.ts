import { NextResponse, type NextRequest } from "next/server";
// import { auth0 } from "./modules/auth/auth0";

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export async function proxy(request: NextRequest) {
  // if (process.env.USE_AUTH0_AUTHORIZATION?.toLowerCase() === "true") {
  //   if (request.nextUrl.pathname === "/auth/token") {
  //       return NextResponse.next();
  //   }

  //   const response: NextResponse = await auth0.middleware(request);

  //   return response;
  // }

  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico, sitemap.xml, robots.txt (metadata files)
     */
    "/((?!_next/static|_next/image|favicon.ico|sitemap.xml|robots.txt).*)",
  ],
};
