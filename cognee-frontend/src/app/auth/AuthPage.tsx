import Link from "next/link";
import { TextLogo } from "@/ui/App";
import { Divider } from "@/ui/Layout";
import Footer from "@/ui/Partials/Footer/Footer";

import { auth0 } from "@/modules/auth/auth0";

import { CTAButton } from "@/ui/elements";
import AuthToken from "./token/AuthToken";


export default async function AuthPage() {
  const session = await auth0.getSession();

  return (
    <main>
      <div className="w-full max-w-md pt-12 pb-6 m-auto">
          <TextLogo width={158} height={44} color="white" />
      </div>
      <Divider />
      <div className="w-full max-w-md pt-12 pb-6 m-auto">
        <div className="flex flex-col w-full gap-8">
          <h1><span className="text-xl">Welcome to cognee</span></h1>
          {session ? (
            <div className="flex flex-col w-full gap-4">
              <span className="text-lg">Hello, {session.user.name}!</span>
              <AuthToken />
              <Link href="/auth/logout">
                <CTAButton>
                  Log out
                </CTAButton>
              </Link>
            </div>
          ) : (
            <>
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
            </>
          )}
        </div>
      </div>
      <Divider />
      <div className="pl-6 pr-6">
        <Footer />
      </div>
    </main>
  )
}
