"use server";

import Dashboard from "./Dashboard";

export default async function Page() {
  const accessToken = "";

  return (
    <Dashboard accessToken={accessToken} />
  );
}
