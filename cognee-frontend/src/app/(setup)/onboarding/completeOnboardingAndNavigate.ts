import { notifications } from "@mantine/notifications";
import { captureException } from "@/utils/monitoring";

let inFlight: Promise<void> | null = null;

/**
 * Shared by every finish/skip action in onboarding. Both setup facts must be
 * durable before navigating: dashboard checks isSeenWelcome before
 * onboardingCompletedAt, so completing only the latter can send a new user
 * back to /welcome while the earlier write is still in flight.
 *
 * A second call while one is running gets the running one back: none of the
 * buttons that call this disables itself, and a double click would otherwise
 * send both writes twice and navigate twice.
 */
export function completeOnboardingAndNavigate(
  markWelcomeSeen: () => Promise<void>,
  markOnboardingComplete: () => Promise<void>,
  navigate: () => void,
): Promise<void> {
  if (inFlight) return inFlight;
  inFlight = complete(markWelcomeSeen, markOnboardingComplete, navigate).finally(() => { inFlight = null; });
  return inFlight;
}

async function complete(
  markWelcomeSeen: () => Promise<void>,
  markOnboardingComplete: () => Promise<void>,
  navigate: () => void,
): Promise<void> {
  try {
    await markWelcomeSeen();
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
