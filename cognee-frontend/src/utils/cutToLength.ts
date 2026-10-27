const HIGH_SURROGATE_START = 0xd800;
const HIGH_SURROGATE_END = 0xdbff;

/**
 * The value cut to at most `max` UTF-16 units, never in the middle of a
 * character. A plain slice can leave the first half of an emoji at the end,
 * and a string like that makes encodeURIComponent throw when it is written
 * into a cookie.
 */
export function cutToLength(value: string, max: number): string {
  if (value.length <= max) return value;
  const cut = value.slice(0, max);
  const last = cut.charCodeAt(cut.length - 1);
  return last >= HIGH_SURROGATE_START && last <= HIGH_SURROGATE_END ? cut.slice(0, -1) : cut;
}
