"use client";

import { useCallback, useRef, useState } from "react";
import { saveAssignment } from "./saveAssignment";

interface AssignmentChoiceOptions {
  /** What is being chosen, as the toasts name it: "Prompt", "Ontology", ... */
  label: string;
  datasetId: string;
  datasetName: string;
  /** The choice as a toast shows it. */
  describe?: (choice: string | null) => string;
  save: (choice: string | null) => Promise<void>;
  /** Runs once a changed choice is stored, e.g. to flag the graph as outdated. */
  onChanged: () => void;
}

interface AssignmentChoice {
  selected: string | null;
  /** True while any save is still in flight. A build started meanwhile would use a choice that is not stored yet. */
  saving: boolean;
  /** Shows the choice at once and stores it behind it. A failed save puts back the last stored choice. */
  select: (choice: string | null) => void;
  /** Takes what the stored configuration says, once it has been read. Never replaces a choice the user has already made. */
  load: (choice: string | null) => void;
  /** Takes a value that is stored by other means (an item was deleted): it replaces the choice and any save still in flight no longer counts. */
  setConfirmed: (choice: string | null) => void;
}

interface PendingSave {
  choice: string | null;
}

/**
 * One of a dataset's choices (graph model, prompt, ontology) that is stored
 * as the user makes it.
 *
 * Saves can overlap, so the newest choice decides what is shown: a save that
 * fails after a newer choice was made leaves the display alone, and only the
 * newest one puts the display back, to the last value that was stored (not the
 * value before the click, which may itself never have been stored). Picking the
 * choice that is already being saved starts no second one.
 *
 * What is shown matches what is stored only because the saves also finish in
 * the order they were made: the configuration writers run one after another per
 * workspace (see the write queue in userConfiguration.ts), so by the time the
 * newest save fails, every older one has settled. A save function without that
 * ordering would need more than this.
 *
 * The configuration is read once the page is up, and the user can pick before
 * that read arrives. `load` therefore only supplies what is stored; it shows it
 * only if no pick is showing, and once a value is known to be stored (a confirmed
 * save, or `setConfirmed`) it knows better than the read, which may have been
 * taken before that.
 */
export function useAssignmentChoice({ label, datasetId, datasetName, describe = (choice) => choice ?? "Automatic", save, onChanged }: AssignmentChoiceOptions): AssignmentChoice {
  const [selected, setSelected] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const stored = useRef<string | null>(null);
  const newest = useRef<PendingSave | null>(null);
  const inFlight = useRef(0);
  const picked = useRef(false);
  const confirmedSave = useRef(false);
  // Bumped when a stored value arrives by other means; saves made before that
  // must not write over it when they finish.
  const generation = useRef(0);

  const load = useCallback((choice: string | null): void => {
    if (confirmedSave.current) return;
    stored.current = choice;
    if (!picked.current) setSelected(choice);
  }, []);

  const setConfirmed = useCallback((choice: string | null): void => {
    generation.current += 1;
    newest.current = null;
    stored.current = choice;
    confirmedSave.current = true;
    setSelected(choice);
  }, []);

  function select(choice: string | null): void {
    const alreadyStored = choice === selected && choice === stored.current;
    const alreadySaving = newest.current !== null && newest.current.choice === choice;
    if (alreadyStored || alreadySaving) return;

    const shownAs = describe(choice);
    const thisSave: PendingSave = { choice };
    const startedIn = generation.current;
    picked.current = true;
    newest.current = thisSave;
    inFlight.current += 1;
    setSelected(choice);
    setSaving(true);
    const isNewest = () => newest.current === thisSave;
    const supersededByStoredValue = () => generation.current !== startedIn;
    void saveAssignment({
      label,
      choice: shownAs,
      datasetId,
      datasetName,
      save: async () => {
        try {
          await save(choice);
        } finally {
          inFlight.current -= 1;
          if (inFlight.current === 0) setSaving(false);
        }
      },
      onSaved: () => {
        if (!supersededByStoredValue()) {
          stored.current = choice;
          confirmedSave.current = true;
        }
        if (isNewest()) newest.current = null;
        onChanged();
      },
      restore: () => {
        if (!isNewest()) return;
        newest.current = null;
        // The display is back on the stored value, so nothing picked is showing
        // any more and a configuration that arrives later may replace it.
        picked.current = false;
        setSelected(stored.current);
      },
    });
  }

  return { selected, saving, select, load, setConfirmed };
}
