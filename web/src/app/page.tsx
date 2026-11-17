import Link from "next/link";
import { ShieldCheck } from "lucide-react";

// Placeholder landing page; the 3D landing page arrives in week 9.
export default function Home() {
  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col justify-center px-6 py-16">
      <div className="flex items-center gap-2 text-accent">
        <ShieldCheck aria-hidden className="size-6" />
        <span className="text-sm font-semibold tracking-wide">FraudGuard</span>
      </div>
      <h1 className="mt-6 text-4xl font-semibold tracking-tight text-ink sm:text-5xl">
        Fraud decisions that weigh the cost of every mistake.
      </h1>
      <p className="mt-4 max-w-xl text-lg text-ink-2">
        Calibrated scores, plain-language reasons and an analyst review queue, on 590,000 real card
        transactions.
      </p>
      <div className="mt-8">
        <Link
          href="/console"
          className="inline-flex h-10 items-center rounded-lg bg-accent px-4 text-sm font-medium text-accent-ink hover:brightness-110"
        >
          Open the analyst console
        </Link>
      </div>
    </main>
  );
}
