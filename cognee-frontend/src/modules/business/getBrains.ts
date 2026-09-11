import type { CogneeInstance } from "@/modules/instances/types";
import type { BrainsPayload } from "./types";

// Every dataset the caller may read, each as a small graph preview — the
// content side of Business when no single brain is focused yet. Already
// group-aware (CLO-399's get_all_user_permission_datasets union), so a
// dataset shared with a group appears here with no extra frontend logic.
export default function getBrains(instance: CogneeInstance): Promise<BrainsPayload> {
  return instance.fetch("/v1/visualize/brains").then((response) => response.json());
}
