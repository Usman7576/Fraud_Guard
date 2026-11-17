import type { Metadata } from "next";

import { THEME_SCRIPT } from "@/components/theme-toggle";

import "./globals.css";

export const metadata: Metadata = {
  title: { default: "FraudGuard", template: "%s · FraudGuard" },
  description: "Cost-aware fraud detection: scores, reasons, review queue and KPIs.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // suppressHydrationWarning: the theme script may set data-theme before React hydrates.
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
