import { notifications } from "@mantine/notifications";
import { captureException } from "@/utils/monitoring";

interface AssignmentSave {
  /** What is being chosen, as the toasts name it: "Prompt", "Ontology", ... */
  label: string;
  /** The choice as the toast shows it. */
  choice: string;
  datasetId: string;
  datasetName: string;
  save: () => Promise<void>;
  /** Runs once the choice is stored, e.g. to flag the graph as outdated. */
  onSaved: () => void;
  /** Undoes this choice on failure. Whoever supplies it decides what to show and must ignore a save that a newer choice has superseded (useAssignmentChoice does). */
  restore: () => void;
}

/**
 * Stores a dataset's graph model, prompt or ontology choice behind a selection
 * that is already showing. The "updated" toast and `onSaved` wait for the save.
 * A failed save calls `restore` and says so: storing needs the current
 * configuration to be read first, and when that read fails nothing is written,
 * so the choice would otherwise look kept and be gone on the next load. A save
 * that times out is ambiguous (the pod may have stored it), so the message does
 * not claim the old value is still there. Always resolves.
 */
export async function saveAssignment({ label, choice, datasetId, datasetName, save, onSaved, restore }: AssignmentSave): Promise<void> {
  try {
    await save();
  } catch (err) {
    captureException(err, { context: "dataset-detail.save-assignment", datasetId, label });
    restore();
    notifications.show({
      title: `${label} not confirmed`,
      message: `Could not confirm that the ${label.toLowerCase()} of "${datasetName}" was changed. Reload the page to see what is stored, or try again.`,
      color: "red",
    });
    return;
  }
  onSaved();
  notifications.show({ title: `${label} updated`, message: `"${datasetName}" now uses "${choice}".`, color: "green", autoClose: 4000 });
}
