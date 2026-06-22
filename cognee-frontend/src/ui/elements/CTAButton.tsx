import classNames from "classnames";
import { ButtonHTMLAttributes } from "react";

export default function CTAButton({ children, className, disabled, ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={classNames(
        "flex flex-row justify-center items-center gap-1.5 rounded-lg bg-cognee-purple px-4 py-2 text-[13px] font-medium text-white leading-4",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cognee-purple",
        {
          "cursor-pointer hover:bg-cognee-purple-hover active:bg-cognee-purple-pressed": !disabled,
          "cursor-not-allowed opacity-60 !bg-cognee-pressed": disabled,
        },
        className,
      )}
      disabled={disabled}
      {...props}
    >
      {children}
    </button>
  );
}
