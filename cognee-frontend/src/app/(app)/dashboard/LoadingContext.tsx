"use client";

import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";

interface LoadingContextValue {
  isLoading: boolean;
  startLoading: () => void;
  stopLoading: () => void;
}

const LoadingContext = createContext<LoadingContextValue>({
  isLoading: true,
  startLoading: () => {},
  stopLoading: () => {},
});

export function LoadingProvider({
  children,
  initialCount = 0,
}: {
  children: React.ReactNode;
  initialCount?: number;
}) {
  const [count, setCount] = useState(initialCount);
  const countRef = useRef(initialCount);

  const startLoading = useCallback(() => {
    countRef.current += 1;
    setCount(countRef.current);
  }, []);

  const stopLoading = useCallback(() => {
    countRef.current = Math.max(0, countRef.current - 1);
    setCount(countRef.current);
  }, []);

  const value = useMemo(
    () => ({ isLoading: count > 0, startLoading, stopLoading }),
    [count, startLoading, stopLoading],
  );

  return (
    <LoadingContext.Provider value={value}>{children}</LoadingContext.Provider>
  );
}

export function useLoading() {
  return useContext(LoadingContext);
}
