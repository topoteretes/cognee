"use server";

import { TrackPageView } from "@/modules/analytics";
import DatasetsBody from "./DatasetsBody";

export default async function Datasets() {
  return (
    <div className="h-full flex flex-col">
      <TrackPageView page="Datasets" />
      <DatasetsBody />
    </div>
  );
}
