import classNames from "classnames";
import { ButtonHTMLAttributes } from "react";

export default function NeutralButton({ children, className, ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button className={classNames("flex flex-row justify-center items-center gap-2 cursor-pointer rounded-3xl bg-transparent px-4 h-8 w-full text-black shadow-xs border-1 border-indigo-600 hover:bg-gray-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600", className)} {...props}>{children}</button>
  );
}
