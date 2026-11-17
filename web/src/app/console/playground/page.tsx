import type { Metadata } from "next";

import { Playground } from "./playground";

export const metadata: Metadata = { title: "Score playground" };

export default function PlaygroundPage() {
  return (
    <div className="mx-auto max-w-6xl space-y-5">
      <header>
        <h1 className="text-xl font-semibold text-ink">Score playground</h1>
        <p className="mt-1 text-sm text-muted">
          Try a transaction and see the decision and its reasons. This is a what-if: nothing is stored and the live
          feature state does not change. It is scored at the latest time the API has seen.
        </p>
      </header>
      <Playground />
    </div>
  );
}
