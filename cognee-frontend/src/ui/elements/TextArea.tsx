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
      className={classNames("block w-full rounded-md bg-white px-4 py-4 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600", className)}
      onChange={handleChange}
      onKeyUp={onKeyUp}
      {...props}
    />
  )
}
