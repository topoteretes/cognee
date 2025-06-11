"use client";

import { useState } from "react";
import { LoadingIndicator } from "@/ui/App";
import { fetch, useBoolean } from "@/utils";
import { CTAButton, Input } from "@/ui/elements";

interface SignInFormPayload extends HTMLFormElement {
  vectorDBUrl: HTMLInputElement;
  vectorDBApiKey: HTMLInputElement;
  llmApiKey: HTMLInputElement;
}

const errorsMap = {
  LOGIN_BAD_CREDENTIALS: "Invalid username or password",
};

export default function SignInForm({ onSignInSuccess = () => window.location.href = "/", submitButtonText = "Sign in" }) {
  const {
    value: isSigningIn,
    setTrue: disableSignIn,
    setFalse: enableSignIn,
  } = useBoolean(false);

  const [signInError, setSignInError] = useState<string | null>(null);

  const signIn = (event: React.FormEvent<SignInFormPayload>) => {
    event.preventDefault();
    const formElements = event.currentTarget;

    const authCredentials = new FormData();
    // Backend expects username and password fields
    authCredentials.append("username", formElements.email.value);
    authCredentials.append("password", formElements.password.value);

    setSignInError(null);
    disableSignIn();

    fetch("/v1/auth/login", {
      method: "POST",
      body: authCredentials,
    })
      .then(() => {
        onSignInSuccess();
      })
      .catch(error => setSignInError(errorsMap[error.detail as keyof typeof errorsMap]))
      .finally(() => enableSignIn());
  };

  return (
    <form onSubmit={signIn} className="flex flex-col gap-2">
      <div className="flex flex-col gap-2">
        <div className="mb-4">
          <label className="block mb-2" htmlFor="email">Email</label>
          <Input id="email" defaultValue="default_user@example.com" name="email" type="email" placeholder="Your email address" />
        </div>
        <div className="mb-4">
          <label className="block mb-2" htmlFor="password">Password</label>
          <Input id="password" defaultValue="default_password" name="password" type="password" placeholder="Your password" />
        </div>
      </div>

      <CTAButton type="submit">
        {submitButtonText}
        {isSigningIn && <LoadingIndicator />}
      </CTAButton>

      {signInError && (
        <span className="text-s text-white">{signInError}</span>
      )}
    </form>
  )
}
