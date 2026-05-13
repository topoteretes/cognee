"use server";

import { TrackPageView } from "@/modules/analytics";
import DashboardBody from "./DashboardBody";

export default async function Dashboard() {
  return (
    <div className="h-full flex flex-col">
      {/* <video
        autoPlay
        loop
        muted
        playsInline
        className="fixed inset-0 z-0 object-cover w-full h-full"
      >
        <source src="/videos/background-video-blur.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video> */}

      <TrackPageView page="Dashboard" />
      <DashboardBody />
    </div>
  );
}
