import type { Metadata } from "next";
import { TextLogo } from "@/ui/App";
import { Divider } from "@/ui/Layout";
import { Footer } from "@/ui/Partials";

export const metadata: Metadata = {
  title: "Cognee",
  description: "Cognee authentication",
};

export default function AuthLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <main className="flex flex-col h-full">
      <div className="pt-6 pr-3 pb-3 pl-6">
        <TextLogo width={86} height={24} />
      </div>
      <Divider />

      {children}

      <Divider />
      <div className="pl-6 pr-6">
        <Footer />
      </div>
    </main>
  );
}
