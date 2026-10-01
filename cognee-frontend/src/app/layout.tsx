import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import "tailwindcss";
import "@mantine/core/styles.css";
import "@mantine/notifications/styles.css";
import { mantineHtmlProps, MantineProvider } from "@mantine/core";
import theme from "@/ui/theme/theme";
import { Notifications } from "@mantine/notifications";
import { OsPreferenceProvider } from "@/ui/layout/OsPreferenceContext";
import QueryProvider from "@/modules/query/QueryProvider";
import RuntimeConfigScript from "@/modules/config/RuntimeConfigScript";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// RuntimeConfigScript below reads COGNEE_BACKEND_URL at render time. Without
// this, the pages that Next can prerender would bake the value in at build
// time and ignore whatever the container was started with.
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Cognee",
  description: "Build AI memory with knowledge graphs.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full" {...mantineHtmlProps}>
      <head>
        <RuntimeConfigScript />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased h-full`}
      >
        <QueryProvider>
          <MantineProvider theme={theme}>
            <Notifications position="top-right" zIndex={10001} />
            <OsPreferenceProvider>
              {children}
            </OsPreferenceProvider>
          </MantineProvider>
        </QueryProvider>
      </body>
    </html>
  );
}
