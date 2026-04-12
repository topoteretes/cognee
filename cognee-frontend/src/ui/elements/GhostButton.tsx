import classNames from "classnames";
import { ButtonHTMLAttributes } from "react";

export default function GhostButton({ children, className, disabled, ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={classNames(
        "flex flex-row justify-center items-center gap-2 rounded-lg bg-transparent px-4 py-2 text-[13px] font-medium text-cognee-body leading-4",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cognee-purple",
        {
          "cursor-pointer hover:bg-cognee-hover active:bg-cognee-pressed": !disabled,
          "cursor-not-allowed opacity-50": disabled,
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
