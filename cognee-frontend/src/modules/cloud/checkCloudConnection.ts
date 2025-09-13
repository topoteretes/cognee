import { fetch } from "@/utils";

export default function checkCloudConnection() {
  return fetch("/v1/checks/connection", {
    method: "POST",
  });
}
