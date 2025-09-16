"use client";

import Link from "next/link";
import { BackIcon } from "@/ui/Icons";
import { CTAButton } from "@/ui/elements";
import Header from "@/ui/Layout/Header";
import { useAuthenticatedUser } from "@/modules/auth";

export default function Account() {
  const { user } = useAuthenticatedUser();
  const account = {
    name: user ? user.name || user.email : "NN",
  };

  return (
    <div className="bg-gray-200 h-full max-w-[1920px] mx-auto">
      <video
        autoPlay
        loop
        muted
        playsInline
        className="fixed inset-0 z-0 object-cover w-full h-full"
      >
        <source src="/videos/background-video-blur.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>

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
    </div>
  );
}
