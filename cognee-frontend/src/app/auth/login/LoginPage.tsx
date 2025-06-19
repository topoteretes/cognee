"use client";

import Link from "next/link";
import Image from "next/image";

import AuthForm from "../AuthForm";

export default function LoginPage() {
  return (
    <div className="m-auto w-full max-w-md shadow-xl rounded-xl">
      <div className="flex flex-col px-10 py-16 bg-white border-1 rounded-xl border-indigo-600 overflow-hidden">
        <Image src="/images/cognee-logo-with-text.png" alt="Cognee logo" width={176} height={46} className="h-12 w-44 self-center mb-16" />

        <h1 className="self-center text-xl mb-4">Welcome</h1>
        <p className="self-center mb-10">Log in to continue with Cognee</p>

        <AuthForm
          authUrl="/v1/auth/login"
          submitButtonText="Login"
          formatPayload={formatPayload}
        />

        <p className="text-center mt-2 text-sm">
          <Link href="/auth/signup">
            {"Or go to Sign up ->"}
          </Link>
        </p>
      </div>
    </div>
  );
}

function formatPayload(data: { email: string, password: string }) {
  const payload = new URLSearchParams();

  payload.append("username", data.email);
  payload.append("password", data.password);

  return payload;
}
