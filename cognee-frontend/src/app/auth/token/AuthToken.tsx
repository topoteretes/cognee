"use client";

import { useEffect } from "react";

export default function AuthToken() {
  useEffect(() => {
    async function get_token() {
      await fetch("http://localhost:3000/auth/token");
    }
    get_token();
  }, []);

  return null;
}
