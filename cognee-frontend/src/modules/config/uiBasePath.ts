/** Prefix raw browser URLs; Next Link/router already apply basePath themselves. */
export function uiPath(path: string): string {
  const base = process.env.NEXT_PUBLIC_UI_BASE_PATH ?? "";
  if (!/^(?:\/[A-Za-z0-9_-]+)*$/.test(base) || !/^\/(?!\/)/.test(path)) {
    throw new Error("Invalid local UI path");
  }
  return base + path;
}
