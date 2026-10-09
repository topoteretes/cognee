export const dynamic = "force-dynamic";

import { redirect } from "next/navigation";
import EmailVerifiedPage from "./EmailVerifiedPage";
import AuthPageContainer from "@/ui/elements/Auth/AuthPageContainer";
import AuthContentSectionCarousel from "@/ui/elements/Auth/ContentSections/AuthContentSectionCarousel";

export default async function Page({
  searchParams,
}: {
  searchParams: Promise<{ verified?: string }>;
}) {
  const params = await searchParams;

  // No verified param — not coming from Auth0 redirect
  if (params.verified !== "true") {
    return redirect("/sign-in");
  }

  return (
    <AuthPageContainer>
      <EmailVerifiedPage />
      <AuthContentSectionCarousel />
    </AuthPageContainer>
  );
}
