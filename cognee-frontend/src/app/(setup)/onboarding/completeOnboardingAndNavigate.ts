import { notifications } from "@mantine/notifications";
import { captureException } from "@/utils/monitoring";

/**
 * Shared by every finish/skip action in onboarding. Navigating on a failed
 * write would leave the dashboard's useOnboardingRedirect reading a stale
 * onboardingCompletedAt and bouncing the user straight back into
 * onboarding, with nothing telling them why — so a failure surfaces here
 * and never navigates, letting the user retry the same click.
 */
export async function completeOnboardingAndNavigate(
  markOnboardingComplete: () => Promise<void>,
  navigate: () => void,
): Promise<void> {
  try {
    await markOnboardingComplete();
    navigate();
  } catch (err) {
    captureException(err, { context: "onboarding-completion" });
    notifications.show({
      color: "red",
      title: "Could not save your progress",
      message: "Something went wrong finishing onboarding. Please try again.",
    });
  }
}
