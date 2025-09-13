import fetch from "@/utils/fetch";

export default function getUser() {
  return fetch("/v1/auth/me")
      .then((response) => response.json());
}
