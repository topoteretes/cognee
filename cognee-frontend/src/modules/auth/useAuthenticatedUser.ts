import { useEffect, useState } from "react";
import { fetch } from "@/utils";
import { User } from "./types";

export default function useAuthenticatedUser() {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    if (!user) {
      fetch("/v1/auth/me")
        .then((response) => response.json())
        .then((data) => setUser(data));
    }
  }, [user]);

  return { user };
}
