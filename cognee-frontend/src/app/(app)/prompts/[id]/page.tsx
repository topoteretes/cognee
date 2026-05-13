export const dynamic = "force-dynamic";

import PromptEditorPage from "./PromptEditorPage";

export default async function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <PromptEditorPage promptId={id} />;
}
