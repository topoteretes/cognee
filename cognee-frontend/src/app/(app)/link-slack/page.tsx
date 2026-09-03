import type { Metadata } from "next";
import LinkSlackPage from "./LinkSlackPage";

export const metadata: Metadata = {
  title: "Connect your Slack account",
};

export default async function Page({
  searchParams,
}: {
  searchParams: Promise<{ code?: string }>;
}) {
  const params = await searchParams;
  return <LinkSlackPage code={params.code ?? ""} />;
}
