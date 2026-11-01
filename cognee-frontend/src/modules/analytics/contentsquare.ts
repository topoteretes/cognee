/**
 * Contentsquare masking marker for the open-source build.
 *
 * The tag and its loader are SaaS-only, but the root layout that carries the
 * marker is synced. Kept as the same class name rather than an empty string:
 * it is inert with no tag to read it, and a class that changes between the
 * two builds is one more way the two layouts drift apart.
 */
export const CS_MASK_CLASS = "cs-mask";
