import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import "tailwindcss";
import "@mantine/core/styles.css";
import "@mantine/notifications/styles.css";
import { mantineHtmlProps, MantineProvider } from "@mantine/core";
import theme from "@/ui/theme/theme";
import { Notifications } from "@mantine/notifications";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

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
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased h-full`}
      >
        <MantineProvider theme={theme}>
          <Notifications position="top-right" zIndex={10001} />
          {children}
        </MantineProvider>
      </body>
    </html>
  );
}
