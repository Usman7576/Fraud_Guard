"use client";

import clsx from "clsx";
import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";

import { StaticGraph } from "./static-art";
import { COLORS, use3DMode, useOnScreen } from "./use-3d";

const GraphScene = dynamic(() => import("./graph-scene"), { ssr: false, loading: () => <StaticGraph /> });

const STEPS = [
  { title: "Every payment leaves links", body: "Each card is tied to the devices and email addresses it pays with." },
  { title: "A chargeback confirms card A", body: "Thirty days later, the bank reports card A's payment as fraud." },
  { title: "One hop: what card A used", body: "Its device and email address are now linked to known fraud." },
  {
    title: "Two hops: the ring",
    body: "Cards B, C and D share that device or email. Their next payments score higher, before any of them is reported.",
  },
] as const;

function stepFor(p: number): number {
  return p < 0.12 ? 0 : p < 0.38 ? 1 : p < 0.64 ? 2 : 3;
}

function Legend() {
  const item = (swatch: React.ReactNode, label: string) => (
    <li className="flex items-center gap-2">
      {swatch}
      {label}
    </li>
  );
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
      {item(<span className="h-2.5 w-4 rounded-sm" style={{ background: COLORS.node }} />, "Card")}
      {item(<span className="size-2.5 rotate-45" style={{ background: COLORS.node }} />, "Device")}
      {item(<span className="size-2.5 rounded-full" style={{ background: COLORS.node }} />, "Email")}
      {item(<span className="size-2.5 rounded-full" style={{ background: COLORS.critical }} />, "Confirmed fraud")}
      {item(<span className="size-2.5 rounded-full" style={{ background: COLORS.warning }} />, "One hop")}
      {item(<span className="size-2.5 rounded-full" style={{ background: COLORS.risk }} />, "Two hops, at risk")}
    </ul>
  );
}

function Intro({ graphRank }: { graphRank: number | null }) {
  return (
    <div>
      <p className="text-sm font-medium text-accent">Entity graph</p>
      <h2 className="mt-2 text-3xl font-semibold tracking-tight text-ink">Fraud travels in rings.</h2>
      <p className="mt-3 text-ink-2">
        The model counts confirmed fraud cards within two hops of every payment, updated live as chargebacks arrive.
        {graphRank !== null && (
          <>
            {" "}
            In the current model that count is feature <span className="tabular font-medium text-ink">#{graphRank}</span> by
            SHAP importance.
          </>
        )}
      </p>
    </div>
  );
}

export function GraphSection({ graphRank }: { graphRank: number | null }) {
  const mode = use3DMode();
  const section = useRef<HTMLElement>(null);
  const progress = useRef(0);
  const [step, setStep] = useState(0);
  const [canvasRef, onScreen] = useOnScreen<HTMLDivElement>();

  useEffect(() => {
    if (mode !== "3d" || !section.current) return;
    let revert: (() => void) | undefined;
    let cancelled = false;
    void Promise.all([import("gsap"), import("gsap/ScrollTrigger")]).then(([{ gsap }, { ScrollTrigger }]) => {
      if (cancelled || !section.current) return;
      gsap.registerPlugin(ScrollTrigger);
      const proxy = { p: 0 };
      const ctx = gsap.context(() => {
        gsap.to(proxy, {
          p: 1,
          ease: "none",
          scrollTrigger: { trigger: section.current, start: "top top", end: "bottom bottom", scrub: 0.8 },
          onUpdate: () => {
            progress.current = proxy.p;
            setStep(stepFor(proxy.p));
          },
        });
      });
      revert = () => ctx.revert();
    });
    return () => {
      cancelled = true;
      revert?.();
    };
  }, [mode]);

  // The tall scroll layout is used while pending too, so the page height does not jump when 3D arrives.
  if (mode === "static") {
    return (
      <section id="graph" className="mx-auto max-w-6xl px-4 py-24 sm:px-6">
        <div className="grid items-center gap-10 lg:grid-cols-[1fr_1.3fr]">
          <div>
            <Intro graphRank={graphRank} />
            <ol className="mt-6 space-y-4">
              {STEPS.map((s, i) => (
                <li key={s.title}>
                  <p className="font-medium text-ink">
                    <span className="tabular mr-2 text-muted">{i + 1}</span>
                    {s.title}
                  </p>
                  <p className="mt-0.5 text-sm text-ink-2">{s.body}</p>
                </li>
              ))}
            </ol>
          </div>
          <div>
            <div className="aspect-[16/10] rounded-2xl border border-line bg-[var(--page)]">
              <StaticGraph />
            </div>
            <div className="mt-3">
              <Legend />
            </div>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section id="graph" ref={section} className="relative h-[320vh]">
      <div className="sticky top-0 flex h-screen flex-col overflow-hidden lg:flex-row">
        <div className="z-10 flex flex-col justify-center px-4 pt-20 sm:px-6 lg:w-[42%] lg:pl-[max(1.5rem,calc((100vw-72rem)/2+1.5rem))] lg:pt-0">
          <Intro graphRank={graphRank} />
          <ol className="mt-6 space-y-3" aria-live="polite">
            {STEPS.map((s, i) => (
              <li
                key={s.title}
                className={clsx(
                  "border-l-2 pl-4 transition-colors duration-300",
                  i === step ? "border-accent" : "border-line",
                  i > step && "hidden lg:block",
                  i < step - 1 && "hidden lg:block",
                )}
              >
                <p className={clsx("font-medium transition-colors", i === step ? "text-ink" : "text-muted")}>{s.title}</p>
                <p className={clsx("mt-0.5 text-sm transition-colors", i === step ? "text-ink-2" : "text-muted/70")}>{s.body}</p>
              </li>
            ))}
          </ol>
          <div className="mt-6 hidden lg:block">
            <Legend />
          </div>
        </div>
        <div ref={canvasRef} className="min-h-0 flex-1">
          {mode === "3d" ? <GraphScene progress={progress} active={onScreen} /> : <StaticGraph />}
        </div>
      </div>
    </section>
  );
}
