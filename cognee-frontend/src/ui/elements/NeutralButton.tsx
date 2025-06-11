import classNames from 'classnames';
import { ButtonHTMLAttributes } from "react";

export default function CTAButton({ children, className, ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button className={classNames("flex flex-row justify-center items-center gap-2 cursor-pointer rounded-md bg-transparent px-3 py-2 text-sm font-semibold text-white shadow-xs border-1 border-white hover:bg-gray-400 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600", className)} {...props}>{children}</button>
  );
}
