"use client";

import { useEffect, useRef, useState } from "react";
import { Avatar, Button, Divider, Flex, Stack, Text, TextInput } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import Image from "next/image";
import { useCurrentUser } from "@/modules/users/useCurrentUser";
import { useUser } from "@/modules/users/UserContext";
import {
  NAME_REFUSED_MESSAGE,
  displayName,
  normalizeNameInput,
  refusedFieldsFromError,
  type ProfileNameField,
  type ProfileNames,
} from "@/modules/users/profileName";
import NameFields, { namesAreSendable, type NameDraft } from "@/ui/elements/NameFields";
import styles from "../SettingsPage.module.css";

function toDraft(names: ProfileNames | null): NameDraft {
  return { firstName: names?.firstName ?? "", lastName: names?.lastName ?? "" };
}

const FIELDS: ProfileNameField[] = ["firstName", "lastName"];

// Whether what is in the form for one name differs from the stored name, the way
// the backend will see them: the same once normalized is not a change.
function isChanged(field: ProfileNameField, shown: NameDraft, stored: ProfileNames | null): boolean {
  return normalizeNameInput(shown[field]) !== normalizeNameInput(stored?.[field]);
}

/**
 * The person's own profile: their name, which they can edit, and their email,
 * which is managed by the sign-in provider (CLO-456).
 *
 * The name shown is the one in the /me cache, and a save puts the backend's
 * normalized form back there, so what the form shows afterwards is what was
 * stored and not what was typed. Nothing is saved until the button is pressed.
 *
 * The draft holds only the names the person has typed into. Every other name is
 * the stored one, so the form never keeps a copy of a name it was not asked to
 * change: a name changed somewhere else (another tab, another device) shows up
 * here, and is not put back to the old value by a save of the other one. A save
 * sends just the names that were edited and differ from the stored ones, and the
 * backend leaves a name that is not sent alone. A name that was emptied is sent
 * as an explicit clear. A successful save drops the draft.
 *
 * The form has three states. While /me is still loading it is disabled and says
 * nothing. If /me has answered without a profile (a backend that predates the
 * field, or a failed request) it is disabled and says editing is unavailable,
 * since an enabled input would write to an endpoint that may not be there.
 * Otherwise it is editable, and an account with no name yet simply has empty
 * inputs.
 */
export default function ProfileWidget() {
  const { data: user } = useCurrentUser();
  const { userMe, isLoading, updateProfileName } = useUser();
  const [draft, setDraft] = useState<Partial<NameDraft>>({});
  const [saving, setSaving] = useState(false);
  // The names the backend refused, kept in the form next to the field rather
  // than in a notification that goes away before it can be read and acted on.
  const [refused, setRefused] = useState<ProfileNameField[]>([]);
  // Moving focus needs the inputs enabled, and they are disabled while saving, so
  // a refusal asks for focus here and an effect honours it once the save is over.
  const [focusRequest, setFocusRequest] = useState<{ field: ProfileNameField; count: number } | null>(null);
  const firstNameRef = useRef<HTMLInputElement>(null);
  const lastNameRef = useRef<HTMLInputElement>(null);

  const stored = userMe?.profile ?? null;
  const editable = stored !== null;
  const unavailable = !editable && !isLoading;
  const shown: NameDraft = { ...toDraft(stored), ...draft };
  const dirty = FIELDS.some((field) => isChanged(field, shown, stored));
  const sendable = namesAreSendable(shown);

  useEffect(() => {
    if (!focusRequest) return;
    (focusRequest.field === "firstName" ? firstNameRef : lastNameRef).current?.focus();
  }, [focusRequest]);

  async function handleSave(): Promise<void> {
    if (stored === null || !dirty || saving || !sendable) return;

    const changes: Partial<ProfileNames> = {};
    for (const field of FIELDS) {
      if (isChanged(field, shown, stored)) changes[field] = normalizeNameInput(shown[field]);
    }

    setSaving(true);
    setRefused([]);
    try {
      await updateProfileName(changes);
      setDraft({});
      notifications.show({
        title: "Profile updated",
        message: "Your name was saved.",
        color: "green",
        autoClose: 4000,
      });
    } catch (e) {
      const fields = refusedFieldsFromError(e);
      if (fields) {
        setRefused(fields);
        setFocusRequest((previous) => ({ field: fields[0], count: (previous?.count ?? 0) + 1 }));
      } else {
        notifications.show({
          color: "red",
          title: "Could not save your name",
          message: "Something went wrong. Please try again.",
        });
      }
    } finally {
      setSaving(false);
    }
  }

  return (
    <Stack
      className={styles.card}
      gap={0}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: "1.375rem" }}>
        <h2 style={{ fontSize: 20, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif' }}>Profile</h2>
        <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0 }}>Your personal account information</p>
      </div>
      <Flex align="center" gap="1rem" mb="1.5rem">
        <Avatar size="lg">
          <Image
            width={54}
            height={54}
            alt=""
            src="/images/icons/avatar.svg"
          />
        </Avatar>
        <Stack gap="0">
          <Text size="sm" fw={700} style={{ color: "#EDECEA" }}>{displayName(stored, user?.name) || "—"}</Text>
          <Text size="xs" style={{ color: "rgba(237,236,234,0.35)" }}>{user?.email ?? "—"}</Text>
        </Stack>
      </Flex>
      <Divider mb="1rem" style={{ borderColor: "rgba(255,255,255,0.08)" }} />
      <form
        onSubmit={(event) => {
          event.preventDefault();
          void handleSave();
        }}
      >
        <Stack gap="0.75rem">
          <NameFields
            idPrefix="profile"
            value={shown}
            onChange={(next) => {
              setRefused([]);
              // Only the input that changed joins the draft; the other keeps
              // following the stored name.
              setDraft((previous) => {
                const field = FIELDS.find((f) => next[f] !== shown[f]);
                return field ? { ...previous, [field]: next[field] } : previous;
              });
            }}
            disabled={!editable || saving}
            errors={{
              firstName: refused.includes("firstName") ? NAME_REFUSED_MESSAGE : undefined,
              lastName: refused.includes("lastName") ? NAME_REFUSED_MESSAGE : undefined,
            }}
            inputRefs={{ firstName: firstNameRef, lastName: lastNameRef }}
          />
          <TextInput
            label="Email"
            value={user?.email ?? ""}
            disabled
            classNames={{ input: "!h-[2.75rem] !border-cognee-border" }}
            radius="0.5rem"
            styles={{
              label: { color: "rgba(237,236,234,0.7)" },
              input: {
                background: "rgba(255,255,255,0.06)",
                borderColor: "rgba(255,255,255,0.12)",
                color: "#EDECEA",
              },
            }}
          />
          <Flex justify="flex-end">
            <Button
              type="submit"
              disabled={!editable || !dirty || !sendable}
              loading={saving}
              radius="0.5rem"
            >
              Save
            </Button>
          </Flex>
        </Stack>
      </form>
      <Text size="sm" style={{ color: "rgba(237,236,234,0.35)" }} mt="1rem">
        {unavailable
          ? "Profile editing is not available right now. Your email is managed by your sign-in provider."
          : "Your email is managed by your sign-in provider."}
      </Text>
    </Stack>
  );
}
