import localFetch from "../instances/localFetch";

export default function checkLocalConnection() {
  return localFetch("/health");
}
