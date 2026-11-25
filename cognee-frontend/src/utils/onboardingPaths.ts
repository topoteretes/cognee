// The new onboarding lives at its own URL while it is built, next to the current
// one at /onboarding, which it does not replace yet. The route renders the
// dashboard tree, so the shell and the tenant-init gate both have to know it. It
// sits here, in synced code, because they do and modules/onboarding is not part
// of the public build.
export const NEW_ONBOARDING_PATH = "/onboarding/window";
