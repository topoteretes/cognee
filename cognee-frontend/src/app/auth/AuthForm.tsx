"use client";

import { useState } from "react";
import { fetch, useBoolean } from "@/utils";
import { CTAButton, Input } from "@/ui/elements";
import { LoadingIndicator } from '@/ui/App';

interface AuthFormPayload extends HTMLFormElement {
  email: HTMLInputElement;
  password: HTMLInputElement;
}

const errorsMap = {
  LOGIN_BAD_CREDENTIALS: "Invalid username or password",
  REGISTER_USER_ALREADY_EXISTS: "User already exists",
};

const defaultFormatPayload: (data: { email: string; password: string; }) => any = (data) => data;

export default function AuthForm({
  submitButtonText = "Sign in",
  authUrl = "/v1/auth/login",
  formatPayload = defaultFormatPayload,
  onSignInSuccess = () => window.location.href = "/",
}) {
  const {
      value: isSigningIn,
      setTrue: disableSignIn,
      setFalse: enableSignIn,
    } = useBoolean(false);

    const [signInError, setSignInError] = useState<string | null>(null);

    const signIn = (event: React.FormEvent<AuthFormPayload>) => {
      event.preventDefault();
      const formElements = event.currentTarget;

      // Backend expects username and password fields
      const authCredentials = {
        email: formElements.email.value,
        password: formElements.password.value,
      };

      setSignInError(null);
      disableSignIn();

      const formattedPayload = formatPayload(authCredentials);

      fetch(authUrl, {
        method: "POST",
        body: formattedPayload instanceof URLSearchParams ? formattedPayload.toString() : JSON.stringify(formattedPayload),
        headers: {
          "Content-Type": formattedPayload instanceof URLSearchParams ? "application/x-www-form-urlencoded" : "application/json",
        },
      })
        .then(() => {
          onSignInSuccess();
        })
        .catch(error => setSignInError(errorsMap[error.detail as keyof typeof errorsMap] || error.message))
        .finally(() => enableSignIn());
    };

    return (
      <form onSubmit={signIn} className="flex flex-col gap-4">
        <label className="flex flex-col gap-1">
          Email address*
          <Input type="email" name="email" required placeholder="Email address*" defaultValue="default_user@example.com" />
        </label>
        <label className="flex flex-col gap-1">
          Password*
          <Input type="password" name="password" required placeholder="Password*" defaultValue="default_password" />
        </label>
        <CTAButton className="mt-6 mb-2" type="submit">
          {submitButtonText}
          {isSigningIn && <LoadingIndicator />}
        </CTAButton>
        {signInError && (
          <span className="text-s text-red-500 mb-4">{signInError}</span>
        )}
      </form>
    );
}
