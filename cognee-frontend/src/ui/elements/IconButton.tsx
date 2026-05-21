import classNames from "classnames";
import { ButtonHTMLAttributes } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  as?: React.ElementType;
}

export default function IconButton({ as, children, className, ...props }: ButtonProps) {
  const Element = as || "button";

  return (
    <Element
      className={classNames(
        "flex flex-row justify-center items-center gap-2 cursor-pointer rounded-lg bg-transparent p-2 -m-2 text-cognee-muted",
        "hover:bg-cognee-hover active:bg-cognee-pressed",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cognee-purple",
        className,
      )}
      {...props}
    >
      {children}
    </Element>
  );
}
