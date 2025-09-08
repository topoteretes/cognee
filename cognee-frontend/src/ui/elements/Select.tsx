import classNames from "classnames";
import { SelectHTMLAttributes } from "react";
import { CaretIcon } from "../Icons";

export default function Select({ children, className, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <div className="relative">
      <select
        className={
          classNames(
            "block w-full appearance-none rounded-3xl bg-white pl-4 pr-8 h-8 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600",
            className,
          )
        }
        {...props}
      >
        {children}
      </select>
      <span className="pointer-events-none absolute top-1/3 -mt-0.5 right-3 text-indigo-600 rotate-180">
        <CaretIcon />
      </span>
    </div>
  );
}
