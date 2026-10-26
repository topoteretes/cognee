import type { ReactNode } from "react";
import { IdentityNotExpected } from "@/modules/analytics";

// These routes are the signed-out half of the app and mount no IdentifyUser,
// so analytics events here must not wait for one. Sign-in and sign-up are also
// where visitors leave fastest, which is what makes the wait expensive rather
// than merely useless.
export default function AuthLayout({ children }: { children: ReactNode }) {
  return (
    <>
      <IdentityNotExpected />
      {children}
    </>
  );
}
