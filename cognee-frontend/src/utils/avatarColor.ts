// Rule: user avatar colours always come from the cognee.ai hero animation
// palette (TetrisCanvas DEFAULT_COLORS on the website landing hero), picked
// deterministically from the user's email/name so a user keeps their colour
// everywhere. All shades are light enough that initials use dark ink.
export const AVATAR_PALETTE = ["#BC9BFF", "#A380EA", "#916DD9", "#DDCCFF", "#E9DFFB", "#F4F4F4", "#9CA3A1"] as const;

export const AVATAR_TEXT = "#1e1e1c";

export function avatarColor(seed: string): string {
  let hash = 0;
  for (let i = 0; i < seed.length; i++) hash = ((hash << 5) - hash + seed.charCodeAt(i)) | 0;
  return AVATAR_PALETTE[Math.abs(hash) % AVATAR_PALETTE.length];
}
