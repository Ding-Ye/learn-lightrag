import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "learn-lightrag",
  description:
    "Learn how LightRAG (HKUDS, EMNLP'25) really works by building a Go mini-version, session by session.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh">
      <body>{children}</body>
    </html>
  );
}
