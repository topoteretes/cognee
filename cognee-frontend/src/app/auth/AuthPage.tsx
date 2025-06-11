import Link from "next/link";

import { auth0 } from "@/modules/auth/auth0";

import { CTAButton } from "@/ui/elements";


export default async function AuthPage() {
  const session = await auth0.getSession();

  return (
  <div className="flex flex-col m-auto max-w-md  h-full gap-8 pb-12 pt-6">
    <h1><span className="text-xl">Welcome to cognee</span></h1>
    {session ? (
      <div className="flex flex-col gap-8">
        <span className="text-lg">Hello, {session.user.name}!</span>
        <Link href="/auth/logout">
          <CTAButton>
            Log out
          </CTAButton>
        </Link>
      </div>
    ) : (
      <div className="flex flex-row h-full gap-8">
        <Link href="/auth/login?screen_hint=signup">
          <CTAButton>
            Sign up
          </CTAButton>
        </Link>

        <Link href="/auth/login">
          <CTAButton>
            Log in
          </CTAButton>
        </Link>
      </div>
    )}
  </div>
  )
}
