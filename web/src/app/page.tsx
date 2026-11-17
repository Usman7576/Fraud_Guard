import { ArrowRight, Database, Github, Scale, ShieldCheck, Trees, Waypoints } from "lucide-react";
import Link from "next/link";

import { GraphSection } from "@/components/landing/graph-section";
import { HeroArt } from "@/components/landing/hero-art";
import { Counter, Motion, Stagger } from "@/components/landing/motion";
import { ScoreDemo } from "@/components/landing/score-demo";
import { GITHUB_URL, fmt, type LandingResults } from "@/lib/landing";
import { loadResults } from "@/lib/results";

function Header() {
  return (
    <header className="fixed inset-x-0 top-0 z-30 border-b border-line/60 bg-[var(--page)]/70 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-6 px-4 sm:px-6">
        <Link href="/" className="flex items-center gap-2 text-ink">
          <ShieldCheck aria-hidden className="size-5 text-accent" />
          <span className="text-sm font-semibold tracking-wide">FraudGuard</span>
        </Link>
        <nav className="hidden gap-5 text-sm text-ink-2 md:flex" aria-label="Sections">
          <a href="#how" className="hover:text-ink">How it works</a>
          <a href="#graph" className="hover:text-ink">Entity graph</a>
          <a href="#demo" className="hover:text-ink">Live demo</a>
          <a href="#results" className="hover:text-ink">Results</a>
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <a href={GITHUB_URL} className="rounded-lg p-2 text-ink-2 hover:text-ink" aria-label="Source code on GitHub">
            <Github aria-hidden className="size-5" />
          </a>
          <Link href="/console" className="rounded-lg border border-line px-3 py-1.5 text-sm text-ink hover:border-accent">
            Sign in
          </Link>
        </div>
      </div>
    </header>
  );
}

function Hero() {
  return (
    <section className="relative flex min-h-[100svh] items-end overflow-hidden pb-16 pt-14 lg:items-center lg:pb-0">
      {/* Phones: the globe fills the top half above the text. Desktop: it sits to the right. */}
      <div className="absolute inset-x-0 top-14 h-[46svh] lg:inset-y-0 lg:left-[38%] lg:right-0 lg:top-0 lg:h-auto" aria-hidden>
        <HeroArt />
      </div>
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-[var(--page)] from-30% to-transparent to-55% lg:bg-gradient-to-r lg:from-[var(--page)] lg:from-0% lg:via-[var(--page)]/70 lg:to-transparent lg:to-100%" />
      <div className="relative mx-auto w-full max-w-6xl px-4 sm:px-6">
        <div className="max-w-xl">
          <p className="text-sm font-medium text-accent">Cost-aware fraud detection</p>
          <h1 className="mt-4 text-4xl font-semibold tracking-tight text-ink sm:text-6xl">
            Catch the fraud.
            <br />
            Keep the customer.
          </h1>
          <p className="mt-5 text-lg text-ink-2">
            FraudGuard scores every card payment in milliseconds, explains each score in plain words, and chooses approve,
            review or decline by what each mistake would cost.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a
              href="#demo"
              className="inline-flex h-11 items-center gap-2 rounded-lg bg-accent px-5 text-sm font-medium text-accent-ink hover:brightness-110"
            >
              Try the live demo <ArrowRight aria-hidden className="size-4" />
            </a>
            <Link
              href="/console"
              className="inline-flex h-11 items-center rounded-lg border border-line bg-[var(--page)]/60 px-5 text-sm font-medium text-ink hover:border-accent"
            >
              Open the analyst console
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}

function HowItWorks({ results }: { results: LandingResults | null }) {
  const p = results?.policy;
  const steps = [
    {
      icon: Database,
      title: "Data",
      body: "590,540 IEEE-CIS card payments, split by time: train on the past, test on a later month. A test fails the build if a future row leaks in.",
    },
    {
      icon: Waypoints,
      title: "Features",
      body: "Velocity windows, spending habits and an entity graph of cards, devices and emails, computed the same way online and offline (parity-tested).",
    },
    {
      icon: Trees,
      title: "Model",
      body: "Tuned LightGBM with isotonic calibration, so a score of 12% means about 12 in 100 are fraud. TreeSHAP turns every score into its top reasons.",
    },
    {
      icon: Scale,
      title: "Decision",
      body: p
        ? `Approve, review or decline by expected cost: the fraud amount against a $${p.falseDeclineCost} false decline and a $${p.reviewCost} review, within ${p.reviewCapacity} reviews a day.`
        : "Approve, review or decline by expected cost: the fraud amount against the cost of a false decline and of a review, within the team's daily review capacity.",
    },
  ];
  return (
    <section id="how" className="mx-auto max-w-6xl scroll-mt-16 px-4 py-24 sm:px-6">
      <p className="text-sm font-medium text-accent">How it works</p>
      <h2 className="mt-2 max-w-2xl text-3xl font-semibold tracking-tight text-ink">From a raw payment to a decision in four steps.</h2>
      <Stagger className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {steps.map((s, i) => (
          <div key={s.title} className="h-full rounded-2xl border border-line bg-surface p-5">
            <div className="flex items-center justify-between">
              <s.icon aria-hidden className="size-5 text-accent" />
              <span className="tabular text-xs text-muted">0{i + 1}</span>
            </div>
            <h3 className="mt-4 font-semibold text-ink">{s.title}</h3>
            <p className="mt-2 text-sm text-ink-2">{s.body}</p>
          </div>
        ))}
      </Stagger>
    </section>
  );
}

function Results({ results }: { results: LandingResults | null }) {
  const card = (label: string, value: React.ReactNode, ci: string, compare: string) => (
    <div className="rounded-2xl border border-line bg-surface p-6">
      <p className="text-sm text-ink-2">{label}</p>
      <p className="mt-2 text-4xl font-semibold tracking-tight text-ink">{value}</p>
      <p className="tabular mt-2 text-xs text-muted">{ci}</p>
      <p className="mt-3 border-t border-line pt-3 text-sm text-ink-2">{compare}</p>
    </div>
  );
  return (
    <section id="results" className="mx-auto max-w-6xl scroll-mt-16 px-4 py-24 sm:px-6">
      <p className="text-sm font-medium text-accent">Results</p>
      <h2 className="mt-2 max-w-2xl text-3xl font-semibold tracking-tight text-ink">Measured on a month the model never saw.</h2>
      {results ? (
        <>
          <div className="mt-10 grid gap-4 md:grid-cols-3">
            {card(
              "Fraud caught at a 1% false-positive rate",
              <Counter value={results.recall.estimate} format="percent" />,
              `95% CI ${fmt(results.recall.low, "percent")} – ${fmt(results.recall.high, "percent")}`,
              `Hand-written rules catch ${fmt(results.rulesRecall, "percent")}.`,
            )}
            {card(
              "Cost saved per 10,000 payments",
              <Counter value={results.costSaved.estimate} format="money" />,
              `95% CI ${fmt(results.costSaved.low, "money")} – ${fmt(results.costSaved.high, "money")}`,
              "Fraud losses, false declines and reviews, against the rules baseline.",
            )}
            {card(
              "PR-AUC",
              <Counter value={results.prAuc.estimate} format="decimal" />,
              `95% CI ${fmt(results.prAuc.low, "decimal")} – ${fmt(results.prAuc.high, "decimal")}`,
              `Rules baseline: ${fmt(results.rulesPrAuc, "decimal")}. Precision and recall over every threshold.`,
            )}
          </div>
          <p className="mt-4 text-sm text-muted">
            Test month of <span className="tabular">{results.nRows.toLocaleString("en-US")}</span> payments (
            <span className="tabular">{results.nFraud.toLocaleString("en-US")}</span> fraud), model v{results.modelVersion}. Intervals
            from a bootstrap over days ({results.nResamples.toLocaleString("en-US")} resamples).
            {results.isSample && " These figures come from the bundled synthetic sample; run the pipeline on IEEE-CIS for the real ones."}
          </p>
        </>
      ) : (
        <p className="mt-6 rounded-2xl border border-line bg-surface p-6 text-sm text-ink-2">
          No evaluation yet. Run <code className="font-mono text-ink">uv run fraudguard evaluate</code> and rebuild the web app to show
          recall, cost saved and PR-AUC here.
        </p>
      )}
    </section>
  );
}

function CallToAction() {
  return (
    <section className="mx-auto max-w-6xl px-4 pb-24 sm:px-6">
      <div className="rounded-3xl border border-line bg-gradient-to-br from-accent-soft to-surface p-8 sm:p-12">
        <h2 className="max-w-xl text-3xl font-semibold tracking-tight text-ink">See the review queue an analyst works from.</h2>
        <p className="mt-3 max-w-xl text-ink-2">
          Cases with reasons, SHAP waterfalls and the entity neighbourhood, weekly KPIs, drift monitoring and model promotion.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link
            href="/console"
            className="inline-flex h-11 items-center gap-2 rounded-lg bg-accent px-5 text-sm font-medium text-accent-ink hover:brightness-110"
          >
            Open the console <ArrowRight aria-hidden className="size-4" />
          </Link>
          <a
            href={GITHUB_URL}
            className="inline-flex h-11 items-center gap-2 rounded-lg border border-line px-5 text-sm font-medium text-ink hover:border-accent"
          >
            <Github aria-hidden className="size-4" /> Read the code
          </a>
        </div>
      </div>
    </section>
  );
}

export default async function Home() {
  const results = await loadResults();
  return (
    <div className="landing min-h-screen">
      <Motion>
        <Header />
        <main>
          <Hero />
          <HowItWorks results={results} />
          <GraphSection graphRank={results?.graphRank ?? null} />
          <section id="demo" className="mx-auto max-w-6xl scroll-mt-16 px-4 py-24 sm:px-6">
            <p className="text-sm font-medium text-accent">Live demo</p>
            <h2 className="mt-2 max-w-2xl text-3xl font-semibold tracking-tight text-ink">Score a payment yourself.</h2>
            <p className="mt-3 max-w-2xl text-ink-2">
              The form calls the real scoring API. The needle shows the calibrated chance of fraud on a log scale; the decision
              comes from the cost policy, not a fixed threshold.
            </p>
            <div className="mt-8">
              <ScoreDemo />
            </div>
          </section>
          <Results results={results} />
          <CallToAction />
        </main>
        <footer className="border-t border-line">
          <div className="mx-auto flex max-w-6xl flex-wrap justify-between gap-2 px-4 py-6 text-xs text-muted sm:px-6">
            <span>Built on the IEEE-CIS Fraud Detection dataset (Vesta). Runs fully local.</span>
            <a href={GITHUB_URL} className="hover:text-ink">
              github.com/Usman7576/Fraud_Guard
            </a>
          </div>
        </footer>
      </Motion>
    </div>
  );
}
