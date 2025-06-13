import classNames from "classnames"
import { InputHTMLAttributes } from "react"

export default function Input({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input className={classNames("block w-full rounded-md bg-white px-4 py-4 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600", className)} {...props} />
  )
}
