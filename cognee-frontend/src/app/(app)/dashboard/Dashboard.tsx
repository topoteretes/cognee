"use server";

import DashboardBody from "./DashboardBody";

export default async function Dashboard() {
  return (
    <div className="h-full flex flex-col">
      <DashboardBody />
    </div>
  );
}
