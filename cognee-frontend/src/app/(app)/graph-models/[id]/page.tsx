"use client";

import { use } from "react";
import GraphModelEditorPage from "./GraphModelEditorPage";

export default function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  return <GraphModelEditorPage modelId={id} />;
}
