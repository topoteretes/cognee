import classNames from 'classnames';
import { ButtonHTMLAttributes } from "react";

export default function CTAButton({ children, className, ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button className={classNames("flex flex-row justify-center items-center gap-2 cursor-pointer rounded-3xl bg-indigo-600 px-4 py-3 text-white hover:bg-indigo-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600", className)} {...props}>{children}</button>
  );
}
