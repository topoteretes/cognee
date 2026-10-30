// The new onboarding lives at its own URL while it is built, next to the current
// one at /onboarding, which it does not replace yet. The route renders the
// dashboard tree, so the shell and the tenant-init gate both have to know it. It
// sits here, in synced code, because they do and modules/onboarding is not part
// of the public build.
export const NEW_ONBOARDING_PATH = "/onboarding/window";

// The current onboarding. The window route sends a visit it refuses here, with
// the hand-over to the window switched off for that visit: the window decides
// from the request's host header and the hand-over from the browser's, and when
// the two ever disagreed the visit would bounce between them for good.
export const OLD_ONBOARDING_PATH = "/onboarding";
export const NO_HANDOVER_PARAM = "handover";
export const NO_HANDOVER_VALUE = "off";
