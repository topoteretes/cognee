/**
 * Open-source stub — the cloud "already seen welcome" redirect check is a
 * cloud-only feature and always dead code here (NEXT_PUBLIC_IS_CLOUD_ENVIRONMENT
 * is forced to "false" for this build).
 */
import WelcomePage from "./WelcomePage";

export default async function WelcomeRoute() {
  return <WelcomePage />;
}
