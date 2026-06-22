import { NextResponse, type NextRequest } from "next/server";

// Local mode — no Auth0 middleware, just pass through
export function middleware(_request: NextRequest) {
  return NextResponse.next();
}

export const config = {
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|sitemap.xml|robots.txt).*)",
  ],
};
