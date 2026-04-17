import classNames from "classnames";
import { SelectHTMLAttributes } from "react";

export default function Select({ children, className, disabled, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <div className="relative">
      <select
        className={classNames(
          "block w-full appearance-none rounded-lg bg-white pl-3.5 pr-8 h-10 text-sm text-cognee-body",
          "border border-cognee-border",
          "hover:bg-cognee-hover",
          "focus:border-cognee-purple focus:border-2 focus:shadow-[0_0_0_3px_rgba(101,16,244,0.1)] focus:outline-none",
          "disabled:bg-cognee-disabled disabled:text-cognee-placeholder disabled:cursor-not-allowed",
          className,
        )}
        disabled={disabled}
        {...props}
      >
        {children}
      </select>
      <span className="pointer-events-none absolute top-1/2 -translate-y-1/2 right-3.5">
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path
            d="M3 4.5L6 7.5L9 4.5"
            stroke={disabled ? "#D4D4D8" : "#A1A1AA"}
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </span>
    </div>
  );
}
