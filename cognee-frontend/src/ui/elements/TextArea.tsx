"use client";

import classNames from "classnames";
import { InputHTMLAttributes, useEffect, useLayoutEffect, useRef } from "react"

interface TextAreaProps extends Omit<InputHTMLAttributes<HTMLTextAreaElement>, "onChange"> {
  isAutoExpanding?: boolean; // Set to true to enable auto-expanding text area behavior. Default is false.
  value: string;
  onChange: (value: string) => void;
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
  const handleTextChange = (event: Event) => {
    const fakeTextAreaElement = event.target as HTMLDivElement;
    const newValue = fakeTextAreaElement.innerText;

    if (newValue !== value) {
      onChange?.(newValue);
    }
  };

  const handleKeyUp = (event: Event) => {
    if (onKeyUp) {
      onKeyUp(event as unknown as React.KeyboardEvent<HTMLTextAreaElement>);
    }
  };

  const handleTextAreaFocus = (event: React.FocusEvent<HTMLDivElement>) => {
    if (event.target.innerText.trim() === placeholder) {
      event.target.innerText = "";
    }
  };
  const handleTextAreaBlur = (event: React.FocusEvent<HTMLDivElement>) => {
    if (value === "") {
      event.target.innerText = placeholder;
    }
  };

  const handleChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(event.target.value);
  };

  const fakeTextAreaRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const fakeTextAreaElement = fakeTextAreaRef.current;

    if (fakeTextAreaElement) {
      fakeTextAreaElement.innerText = placeholder;
      fakeTextAreaElement.addEventListener("input", handleTextChange);
      fakeTextAreaElement.addEventListener("keyup", handleKeyUp);
    }

    return () => {
      if (fakeTextAreaElement) {
        fakeTextAreaElement.removeEventListener("input", handleTextChange);
        fakeTextAreaElement.removeEventListener("keyup", handleKeyUp);
      }
    };
  }, []);

  useEffect(() => {
    const fakeTextAreaElement = fakeTextAreaRef.current;
    const textAreaText = fakeTextAreaElement?.innerText;
    if (fakeTextAreaElement && textAreaText !== value && textAreaText !== placeholder) {
      fakeTextAreaElement.innerText = value;
    }
  }, [value]);

  return isAutoExpanding ? (
    <>
      <div
        ref={fakeTextAreaRef}
        contentEditable="true"
        role="textbox"
        aria-multiline="true"
        className={classNames("block w-full rounded-md bg-white px-4 py-4 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600", className)}
        onFocus={handleTextAreaFocus}
        onBlur={handleTextAreaBlur}
      />
    </>
  ) : (
    <textarea
      name={name}
      style={style}
      value={value}
      placeholder={placeholder}
      className={classNames("block w-full rounded-md bg-white px-4 py-4 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600", className)}
      onChange={handleChange}
      {...props}
    />
  )
}
