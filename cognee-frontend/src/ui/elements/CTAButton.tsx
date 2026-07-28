import classNames from "classnames";
import { ButtonHTMLAttributes } from "react";

export default function CTAButton({ children, className, disabled, ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={classNames(
        "flex flex-row justify-center items-center gap-1.5 rounded-lg bg-[#6510F4] px-4 py-2 text-[13px] font-medium text-white leading-4",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#6510F4]",
        {
          "cursor-pointer hover:bg-[#5A0ED6] active:bg-[#4A0BAF]": !disabled,
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
