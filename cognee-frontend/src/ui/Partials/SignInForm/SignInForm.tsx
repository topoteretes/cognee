"use client";

import {
  CTAButton,
  FormGroup,
  FormInput,
  FormLabel,
  Input,
  Spacer,
  Stack,
  Text,
  useBoolean,
} from 'ohmy-ui';
import { LoadingIndicator } from '@/ui/App';
import { fetch, handleServerErrors } from '@/utils';
import { useState } from 'react';

interface SignInFormPayload extends HTMLFormElement {
  vectorDBUrl: HTMLInputElement;
  vectorDBApiKey: HTMLInputElement;
  llmApiKey: HTMLInputElement;
}

const errorsMap = {
  LOGIN_BAD_CREDENTIALS: 'Invalid username or password',
};

export default function SignInForm({ onSignInSuccess = () => window.location.href = '/', submitButtonText = 'Sign in' }) {
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

    fetch('/v1/auth/login', {
      method: 'POST',
      body: authCredentials,
    })
      .then(handleServerErrors)
      .then(response => response.json())
      .then((bearer) => {
        window.localStorage.setItem('access_token', bearer.access_token);
        onSignInSuccess();
      })
      .catch(error => setSignInError(errorsMap[error.detail as keyof typeof errorsMap]))
      .finally(() => enableSignIn());
  };

  return (
    <form onSubmit={signIn} style={{ width: '100%' }}>
      <Stack gap="4" orientation="vertical">
        <Stack gap="4" orientation="vertical">
          <FormGroup orientation="vertical" align="center/" gap="2">
            <FormLabel>Email:</FormLabel>
            <FormInput>
              <Input name="email" type="email" placeholder="Your email address" />
            </FormInput>
          </FormGroup>
          <FormGroup orientation="vertical" align="center/" gap="2">
            <FormLabel>Password:</FormLabel>
            <FormInput>
              <Input name="password" type="password" placeholder="Your password" />
            </FormInput>
          </FormGroup>
        </Stack>

        <Spacer top="2">
          <CTAButton type="submit">
            <Stack gap="2" orientation="horizontal" align="/center">
              {submitButtonText}
              {isSigningIn && <LoadingIndicator />}
            </Stack>
          </CTAButton>
        </Spacer>

        {signInError && (
          <Text>{signInError}</Text>
        )}
      </Stack>
    </form>
  )
}
