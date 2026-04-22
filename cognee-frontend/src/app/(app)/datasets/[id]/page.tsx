"use client";

import { use } from "react";
import DatasetDetailPage from "./DatasetDetailPage";

export default function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  return <DatasetDetailPage datasetId={id} />;
}
