import { notifications } from "@mantine/notifications";

const SUCCESS_AUTO_CLOSE_MS = 4000;
const WARNING_AUTO_CLOSE_MS = 6000;

export function notifySuccess(title: string, message: string): void {
  notifications.show({ title, message, color: "green", autoClose: SUCCESS_AUTO_CLOSE_MS });
}

export function notifyWarning(title: string, message: string): void {
  notifications.show({ title, message, color: "yellow", autoClose: WARNING_AUTO_CLOSE_MS });
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** A refused change, with the server's own reason when it gave one. */
export function notifyFailure(title: string, error: unknown): void {
  const message = errorMessage(error);
  console.error(`[MEMBERS] ${title}:`, message);
  notifications.show({ title, message: message || "Please try again.", color: "red" });
}
