"use client";

import { useCallback, useEffect, useRef } from "react";

/**
 * Whether the component is still on screen, for code that continues after an
 * await. A navigation issued then lands on whatever page the user has gone to
 * in the meantime and pulls them back (CLO-1110), so it is skipped once the
 * component is gone. The effect sets the flag again on mount so a StrictMode
 * remount does not leave it false.
 */
export function useIsMounted(): () => boolean {
  const mounted = useRef(false);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  return useCallback(() => mounted.current, []);
}
