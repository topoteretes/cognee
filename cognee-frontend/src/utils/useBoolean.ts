import { useCallback, useState } from "react";

export default function useBoolean(initialValue: boolean) {
  const [value, setValue] = useState(initialValue);

  const setTrue = useCallback(() => setValue(true), []);
  const setFalse = useCallback(() => setValue(false), []);
  const toggle = useCallback(() => setValue((prevValue) => !prevValue), []);

  return {
    value,
    setTrue,
    setFalse,
    toggle,
  };
}
