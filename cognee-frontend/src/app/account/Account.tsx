import Link from "next/link";
import { BackIcon } from "@/ui/Icons";
import { CTAButton } from "@/ui/elements";
import Header from "@/ui/Layout/Header";

export default function Account() {
  const account = {
    name: "John Doe",
  };

  return (
    <>
      <div className="absolute top-0 right-0 bottom-0 left-0 flex flex-row gap-2.5">
        <div className="flex-1/5 bg-gray-100 h-full"></div>
        <div className="flex-1/5 bg-gray-100 h-full"></div>
        <div className="flex-1/5 bg-gray-100 h-full"></div>
        <div className="flex-1/5 bg-gray-100 h-full"></div>
        <div className="flex-1/5 bg-gray-100 h-full"></div>
      </div>

      <Header />

      <div className="relative flex flex-row items-start gap-2.5">
        <Link href="/dashboard" className="flex-1/5 py-4 px-5 flex flex-row items-center gap-5">
          <BackIcon />
          <span>back</span>
        </Link>
        <div className="flex-1/5 flex flex-col gap-2.5">
          <div className="py-4 px-5 rounded-xl bg-white">
            <div>Account</div>
            <div className="text-sm text-gray-400 mb-8">Manage your account&apos;s settings.</div>
            <div>{account.name}</div>
          </div>
          <div className="py-4 px-5 rounded-xl bg-white">
            <div>Plan</div>
            <div className="text-sm text-gray-400 mb-8">You are using open-source version. Subscribe to get access to hosted cognee with your data!</div>
            <Link href="/plan">
              <CTAButton><span className="">Select a plan</span></CTAButton>
            </Link>
          </div>
        </div>
        <div className="flex-1/5 py-4 px-5 rounded-xl">
        </div>
        <div className="flex-1/5 py-4 px-5 rounded-xl">
        </div>
        <div className="flex-1/5 py-4 px-5 rounded-xl">
        </div>
      </div>
    </>
  );
}
