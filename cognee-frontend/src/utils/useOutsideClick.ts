import { useEffect, useRef } from "react";

export default function useOutsideClick<ElementType extends HTMLElement>(callbackFn: () => void, isEnabled = true) {
  const rootElementRef = useRef<ElementType>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      const clickedElement = event.target;

      if (clickedElement && rootElementRef.current && !rootElementRef.current?.contains(clickedElement as Node)) {
        callbackFn();
      }
    }

    if (isEnabled) {
      document.addEventListener("click", handleClickOutside);

      return () => {
        document.removeEventListener("click", handleClickOutside);
      };
    }
  }, [callbackFn, isEnabled]);

  return rootElementRef;
}
