import classNames from "classnames"
import { InputHTMLAttributes } from "react"

export default function Input({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={classNames(
        "block w-full rounded-lg bg-white px-3.5 h-10 text-sm text-cognee-body",
        "border border-cognee-border",
        "placeholder:text-cognee-placeholder",
        "hover:bg-cognee-hover",
        "focus:border-[#6510F4] focus:border-2 focus:shadow-[0_0_0_3px_rgba(188,155,255,0.10)] focus:outline-none",
        "disabled:bg-cognee-disabled disabled:text-cognee-placeholder disabled:cursor-not-allowed",
        className,
      )}
      {...props}
    />
  )
}
