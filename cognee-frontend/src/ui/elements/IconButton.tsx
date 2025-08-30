import classNames from "classnames";
import { ButtonHTMLAttributes } from "react";

export default function IconButton({ children, className, ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button className={classNames("flex flex-row justify-center items-center gap-2 cursor-pointer rounded-xl bg-transparent p-[0.5rem] m-[-0.5rem] text-black hover:bg-gray-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600", className)} {...props}>{children}</button>
  );
}
