"use client";

import classNames from "classnames";
import { InputHTMLAttributes, useCallback, useEffect, useRef } from "react"

interface TextAreaProps extends Omit<InputHTMLAttributes<HTMLTextAreaElement>, "onChange"> {
  isAutoExpanding?: boolean; // Set to true to enable auto-expanding text area behavior. Default is false.
  value?: string;
  onChange?: (value: string) => void;
}

export default function TextArea({
  isAutoExpanding,
  style,
  name,
  value,
  onChange,
  className,
  placeholder = "",
  onKeyUp,
  ...props
}: TextAreaProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const maxHeightRef = useRef<number | null>(null);
  const throttleTimeoutRef = useRef<number | null>(null);
  const lastAdjustTimeRef = useRef<number>(0);
  const THROTTLE_MS = 250; // 4 calculations per second

  const adjustHeight = useCallback(() => {
    if (!isAutoExpanding || !textareaRef.current) return;

    const textarea = textareaRef.current;

    // Cache maxHeight on first calculation
    if (maxHeightRef.current === null) {
      const computedStyle = getComputedStyle(textarea);
      maxHeightRef.current = computedStyle.maxHeight === "none"
        ? Infinity
        : parseInt(computedStyle.maxHeight) || Infinity;
    }

    // Reset height to auto to get the correct scrollHeight
    textarea.style.height = "auto";
    // Set height to scrollHeight, but respect max-height
    const scrollHeight = textarea.scrollHeight;
    textarea.style.height = `${Math.min(scrollHeight, maxHeightRef.current)}px`;
    lastAdjustTimeRef.current = Date.now();
  }, [isAutoExpanding]);

  const handleChange = useCallback((event: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = event.target.value;
    onChange?.(newValue);

    // Throttle height adjustments to avoid blocking typing
    if (isAutoExpanding) {
      const now = Date.now();
      const timeSinceLastAdjust = now - lastAdjustTimeRef.current;

      if (timeSinceLastAdjust >= THROTTLE_MS) {
        adjustHeight();
      } else {
        if (throttleTimeoutRef.current !== null) {
          clearTimeout(throttleTimeoutRef.current);
        }
        throttleTimeoutRef.current = window.setTimeout(() => {
          adjustHeight();
          throttleTimeoutRef.current = null;
        }, THROTTLE_MS - timeSinceLastAdjust);
      }
    }
  }, [onChange, isAutoExpanding, adjustHeight]);

  useEffect(() => {
    if (isAutoExpanding && textareaRef.current) {
      adjustHeight();
    }
  }, [value, isAutoExpanding, adjustHeight]);

  useEffect(() => {
    return () => {
      if (throttleTimeoutRef.current !== null) {
        clearTimeout(throttleTimeoutRef.current);
      }
    };
  }, []);

  return (
    <textarea
      ref={isAutoExpanding ? textareaRef : undefined}
      name={name}
      style={style}
      value={value}
      placeholder={placeholder}
      className={classNames(
        "block w-full rounded-lg bg-white px-3.5 py-3 text-sm text-cognee-body",
        "border border-cognee-border",
        "placeholder:text-cognee-placeholder",
        "hover:bg-cognee-hover",
        "focus:border-cognee-purple focus:border-2 focus:shadow-[0_0_0_3px_rgba(101,16,244,0.1)] focus:outline-none",
        className,
      )}
      onChange={handleChange}
      onKeyUp={onKeyUp}
      {...props}
    />
  )
}
