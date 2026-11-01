import { redirect } from "next/navigation";
import isCloudEnvironment from "@/utils/isCloudEnvironment";
import OnboardingPage from "./OnboardingPage";

export default async function OnboardingRoute() {
  if (!isCloudEnvironment()) redirect("/dashboard");
  return <OnboardingPage />;
}
