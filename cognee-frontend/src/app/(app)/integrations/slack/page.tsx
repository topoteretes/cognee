import type { Metadata } from "next";
import SlackGuide from "./SlackGuide";

export const metadata: Metadata = {
  title: "How Slack memory works",
};

export default async function Page() {
  return <SlackGuide />;
}
