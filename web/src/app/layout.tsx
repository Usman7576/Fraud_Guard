import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: { default: "FraudGuard", template: "%s · FraudGuard" },
  description: "Cost-aware fraud detection: scores, reasons, review queue and KPIs.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
