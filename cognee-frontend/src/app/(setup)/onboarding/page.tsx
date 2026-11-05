/**
 * Open-source stub — the cloud "already onboarded" redirect check is a
 * cloud-only feature and always dead code here (NEXT_PUBLIC_IS_CLOUD_ENVIRONMENT
 * is forced to "false" for this build).
 */
import OnboardingPage from "./OnboardingPage";

export default async function OnboardingRoute() {
  return <OnboardingPage />;
}
