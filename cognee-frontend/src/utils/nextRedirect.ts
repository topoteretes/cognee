export const NEXT_REDIRECT = "NEXT_REDIRECT";

export function isNextRedirect(e: unknown): boolean {
  return e instanceof Error && e.message === NEXT_REDIRECT;
}
