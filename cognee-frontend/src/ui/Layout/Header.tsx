import Link from "next/link";
import Image from "next/image";
import { CloudIcon, CogneeIcon, SettingsIcon } from "../Icons";

export default function Header() {
  return (
    <header className="relative bg-[rgba(244,244,244,0.5)] flex flex-row h-14 px-5 items-center justify-between w-full max-w-[1920px] mx-auto">
      <div className="flex flex-row gap-4 items-center">
        <CogneeIcon />
        <div className="text-lg">Cognee Graph Interface</div>
      </div>

      <div className="flex flex-row items-center">
        <div className="flex flex-row gap-3 items-center pl-6 pr-4">
          <CloudIcon />
          <div className="text-indigo-700">Sync</div>
        </div>
        <div className="text-indigo-700 pl-4 pr-6">Premium</div>
        <div className="px-2 py-2 mr-3">
          <SettingsIcon />
        </div>
        <Link href="/account" className="bg-indigo-600 w-8 h-8 rounded-full overflow-hidden">
          <Image width="32" height="32" alt="Name of the user" src="/images/cognee-logo-with-text.png" />
        </Link>
      </div>
    </header>
  );
}
