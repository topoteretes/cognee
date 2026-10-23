import { HttpError, isApiErrorBody } from "@/services/http/errors";

/**
 * A person's first and last name (CLO-456): what the client does with one before
 * it sends it, and the contract between the backend, the /api/me/profile route
 * and the profile form for a name that is refused.
 *
 * The backend is the authority. It normalizes (NFC, invisible characters
 * stripped) and refuses what no name contains, and its 422 is the final word.
 * This module only does what a form needs to be pleasant and honest: normalize
 * to the same form the backend does where that is cheap (NFC, whitespace), so
 * that "unchanged" and "too long" mean the same here as there, and stop an
 * over-long value before the round trip. It must not grow into a second copy of
 * the backend's rules: two copies of a security policy drift, and the one that
 * drifts is the client's.
 */

/** Mirrors the backend's MAX_NAME_LENGTH (a VARCHAR(100)); counted in code points. */
export const NAME_MAX_LENGTH = 100;

/**
 * Which surface wrote the name; the backend stores it as provenance. Settings is
 * the only one this client sends: a name is edited on the profile and nowhere
 * else, and in particular is not part of onboarding.
 */
export type ProfileNameSource = "settings";

export function isProfileNameSource(value: unknown): value is ProfileNameSource {
  return value === "settings";
}

export type ProfileNameField = "firstName" | "lastName";

export interface ProfileNames {
  firstName: string | null;
  lastName: string | null;
}

/**
 * The stored form of what a person typed: NFC, whitespace collapsed, ends
 * trimmed, and empty is null. NFC first, so that a name pasted in decomposed form
 * (macOS file names, some input methods, Hangul) is the same name, and the same
 * length, as the composed one the backend will store.
 */
export function normalizeNameInput(raw: string | null | undefined): string | null {
  const collapsed = (raw ?? "").normalize("NFC").trim().replace(/\s+/g, " ");
  return collapsed === "" ? null : collapsed;
}

/** Length the way the backend counts it, in code points and not UTF-16 units. */
function codePointLength(value: string): number {
  return Array.from(value).length;
}

export type NameProblem = "too_long";

/** The reason a name cannot be sent, or null. Empty is not a problem: it clears. */
export function nameProblem(raw: string | null | undefined): NameProblem | null {
  const name = normalizeNameInput(raw);
  if (name !== null && codePointLength(name) > NAME_MAX_LENGTH) return "too_long";
  return null;
}

export const NAME_PROBLEM_MESSAGE: Record<NameProblem, string> = {
  too_long: `Use at most ${NAME_MAX_LENGTH} characters.`,
};

/** What the backend's refusal means to a person. It names no character on purpose. */
export const NAME_REFUSED_MESSAGE =
  "That name has characters we can't accept. Try typing it again.";

/** Whether two sets of names are the same once normalized, for "has anything changed". */
export function sameNames(a: ProfileNames, b: ProfileNames): boolean {
  return (
    normalizeNameInput(a.firstName) === normalizeNameInput(b.firstName) &&
    normalizeNameInput(a.lastName) === normalizeNameInput(b.lastName)
  );
}

/** The name to show for a person: their own first and last name, else the fallback. */
export function displayName(names: ProfileNames | null, fallback: string | null | undefined): string {
  const own = [names?.firstName, names?.lastName].filter(Boolean).join(" ");
  return own || fallback || "";
}

// ── A refused name: backend 422 -> route -> form ─────────────────────────────

/** The `error` the route answers with when the backend refused a name. */
export const NAME_REFUSED = "name_refused";

const FIELD_BY_LOCATION: Record<string, ProfileNameField> = {
  firstName: "firstName",
  first_name: "firstName",
  lastName: "lastName",
  last_name: "lastName",
};

/**
 * Which names a backend 422 refused, from FastAPI's validation `detail`: a list
 * of {loc: ["body", "<field>"], msg, input}. Only the field names are taken. The
 * entry's `input` is the name the person typed, and neither the route nor the
 * client has any business passing it on.
 *
 * The field is named in camelCase (the alias) or snake_case depending on how the
 * backend reports it, so both are read. Anything else is ignored, and an empty
 * result means the backend refused the request without saying which name.
 */
export function refusedFieldsFromDetail(detail: unknown): ProfileNameField[] {
  if (!Array.isArray(detail)) return [];
  const fields = new Set<ProfileNameField>();
  for (const entry of detail) {
    const loc: unknown = entry && typeof entry === "object" ? (entry as { loc?: unknown }).loc : undefined;
    const last = Array.isArray(loc) ? loc[loc.length - 1] : undefined;
    const field = typeof last === "string" ? FIELD_BY_LOCATION[last] : undefined;
    if (field) fields.add(field);
  }
  return [...fields];
}

/**
 * Which names the route said were refused, or null when this error is not a
 * refused name. When the route could not say which, both are returned: the
 * message under each is true of whichever it was, which a message under only one
 * would not be.
 */
export function refusedFieldsFromError(error: unknown): ProfileNameField[] | null {
  if (!(error instanceof HttpError) || error.status !== 422) return null;
  const body = error.body;
  if (!isApiErrorBody(body) || body.error !== NAME_REFUSED) return null;
  const named = Array.isArray(body.fields)
    ? body.fields.filter((f): f is ProfileNameField => f === "firstName" || f === "lastName")
    : [];
  return named.length > 0 ? named : ["firstName", "lastName"];
}
