"use client";

import type { Ref } from "react";
import { Flex, TextInput } from "@mantine/core";
import {
  NAME_PROBLEM_MESSAGE,
  nameProblem,
  type ProfileNameField,
} from "@/modules/users/profileName";

const INPUT_STYLES = {
  label: { color: "rgba(237,236,234,0.7)" },
  input: {
    background: "rgba(255,255,255,0.06)",
    borderColor: "rgba(255,255,255,0.12)",
    color: "#EDECEA",
  },
} as const;

export interface NameDraft {
  firstName: string;
  lastName: string;
}

interface Props {
  // What is in the two inputs, exactly as typed. The caller normalizes when it
  // sends; showing the normalized form while someone types would eat spaces.
  value: NameDraft;
  onChange: (next: NameDraft) => void;
  disabled?: boolean;
  // A message to show under a field, for a refusal the backend made. A length
  // problem is the component's own and takes precedence, since it is the one
  // the person can see for themselves.
  errors?: Partial<Record<ProfileNameField, string>>;
  // So the caller can move focus to the field an error is about.
  inputRefs?: Partial<Record<ProfileNameField, Ref<HTMLInputElement>>>;
  idPrefix: string;
}

/** True when the typed values can be sent: nothing over the length limit. */
export function namesAreSendable(value: NameDraft): boolean {
  return nameProblem(value.firstName) === null && nameProblem(value.lastName) === null;
}

function errorFor(
  field: ProfileNameField,
  value: NameDraft,
  errors: Props["errors"],
): string | undefined {
  const problem = nameProblem(value[field]);
  return problem ? NAME_PROBLEM_MESSAGE[problem] : errors?.[field];
}

/**
 * The first and last name inputs. Presentational: it owns no request and no
 * state, so the caller decides when a save happens.
 */
export default function NameFields({
  value,
  onChange,
  disabled = false,
  errors,
  inputRefs,
  idPrefix,
}: Props): React.JSX.Element {
  return (
    <Flex gap="0.75rem" wrap="wrap">
      <TextInput
        id={`${idPrefix}-first-name`}
        ref={inputRefs?.firstName}
        label="First name"
        placeholder="First name"
        autoComplete="given-name"
        value={value.firstName}
        onChange={(event) => onChange({ ...value, firstName: event.currentTarget.value })}
        disabled={disabled}
        // No maxLength on the input: it would silently cut a pasted name, and
        // the limit is on the normalized name, not on what is typed.
        error={errorFor("firstName", value, errors)}
        style={{ flex: "1 1 10rem" }}
        classNames={{ input: "!h-[2.75rem] !border-cognee-border" }}
        radius="0.5rem"
        styles={INPUT_STYLES}
      />
      <TextInput
        id={`${idPrefix}-last-name`}
        ref={inputRefs?.lastName}
        label="Last name"
        placeholder="Last name"
        autoComplete="family-name"
        value={value.lastName}
        onChange={(event) => onChange({ ...value, lastName: event.currentTarget.value })}
        disabled={disabled}
        error={errorFor("lastName", value, errors)}
        style={{ flex: "1 1 10rem" }}
        classNames={{ input: "!h-[2.75rem] !border-cognee-border" }}
        radius="0.5rem"
        styles={INPUT_STYLES}
      />
    </Flex>
  );
}
