"use client";

// Charts for the console. Single-series, so no legend box: the card title names the measure.
// Specs: 2px lines, a 10% area wash, hairline solid gridlines, an end value label, and a
// crosshair + tooltip on hover or keyboard focus. Text always uses ink tokens, never series colour.
import { useEffect, useRef, useState } from "react";

export interface Point {
  label: string; // x label (e.g. week start)
  value: number | null;
}

function niceMax(v: number): number {
  if (v <= 0) return 1;
  const exp = 10 ** Math.floor(Math.log10(v));
  const f = v / exp;
  const nice = f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10;
  return nice * exp;
}

function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([entry]) => entry && setWidth(entry.contentRect.width));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

export function LineChart({
  data,
  format,
  height = 150,
  ariaLabel,
  reference,
}: {
  data: Point[];
  format: (v: number) => string;
  height?: number;
  ariaLabel: string;
  /** A labelled horizontal line, e.g. a retraining threshold. */
  reference?: { value: number; label: string };
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const pad = { top: 12, right: 56, bottom: 22, left: 44 };
  const w = Math.max(width - pad.left - pad.right, 10);
  const h = height - pad.top - pad.bottom;
  const values = data.map((d) => d.value).filter((v): v is number => v !== null);
  const max = niceMax(Math.max(...values, reference?.value ?? 0, 0));
  const x = (i: number) => pad.left + (data.length <= 1 ? w / 2 : (i / (data.length - 1)) * w);
  const y = (v: number) => pad.top + h - (v / max) * h;
  const ticks = [0, max / 2, max];

  // Line segments skip missing values (null) rather than drawing through them.
  const segments: [number, number][][] = [];
  let current: [number, number][] = [];
  data.forEach((d, i) => {
    if (d.value === null) {
      if (current.length) segments.push(current);
      current = [];
    } else current.push([x(i), y(d.value)]);
  });
  if (current.length) segments.push(current);
  const linePath = (pts: [number, number][]) =>
    pts.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`).join("");
  const base = pad.top + h;
  const areaPath = (pts: [number, number][]) =>
    `${linePath(pts)}L${pts.at(-1)![0].toFixed(1)},${base}L${pts[0]![0].toFixed(1)},${base}Z`;
  const lastIdx = data.findLastIndex((d) => d.value !== null);
  const last = lastIdx >= 0 ? data[lastIdx] : undefined;

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left - pad.left;
    const i = data.length <= 1 ? 0 : Math.round((px / w) * (data.length - 1));
    setHover(Math.min(Math.max(i, 0), data.length - 1));
  };
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowRight") setHover((i) => Math.min((i ?? -1) + 1, data.length - 1));
    if (e.key === "ArrowLeft") setHover((i) => Math.max((i ?? data.length) - 1, 0));
    if (e.key === "Escape") setHover(null);
  };
  const hp = hover !== null ? data[hover] : undefined;

  return (
    <div ref={ref} className="relative w-full">
      {width > 0 && (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={ariaLabel}
          tabIndex={0}
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
          onKeyDown={onKey}
          onBlur={() => setHover(null)}
          className="block touch-none outline-none"
        >
          {ticks.map((t) => (
            <g key={t}>
              <line x1={pad.left} x2={pad.left + w} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeWidth={1} />
              <text x={pad.left - 8} y={y(t)} dy="0.32em" textAnchor="end" className="tabular fill-[var(--muted)] text-[10px]">
                {format(t)}
              </text>
            </g>
          ))}
          {data.map((d, i) =>
            i === 0 ||
            i === data.length - 1 ||
            // Interior labels every ~fifth of the axis, but never crowding the last one.
            (data.length > 6 && i % Math.ceil(data.length / 5) === 0 && data.length - 1 - i >= Math.ceil(data.length / 5) / 2) ? (
              <text key={d.label + i} x={x(i)} y={height - 6} textAnchor="middle" className="fill-[var(--muted)] text-[10px]">
                {d.label}
              </text>
            ) : null,
          )}
          {reference && (
            <g>
              <line
                x1={pad.left}
                x2={pad.left + w}
                y1={y(reference.value)}
                y2={y(reference.value)}
                stroke="var(--critical)"
                strokeWidth={1}
                opacity={0.7}
              />
              <text x={pad.left + w + 4} y={y(reference.value)} dy="0.32em" className="fill-[var(--ink-2)] text-[10px]">
                {reference.label}
              </text>
            </g>
          )}
          {segments.map((seg) => (
            <g key={linePath(seg)}>
              {seg.length > 1 && <path d={areaPath(seg)} fill="var(--accent)" opacity={0.1} />}
              <path
                d={linePath(seg)}
                fill="none"
                stroke="var(--accent)"
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            </g>
          ))}
          {last && last.value !== null && hover === null && (
            <g>
              <circle cx={x(lastIdx)} cy={y(last.value)} r={4} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />
              <text x={x(lastIdx) + 8} y={y(last.value)} dy="0.32em" className="tabular fill-[var(--ink)] text-[11px] font-medium">
                {format(last.value)}
              </text>
            </g>
          )}
          {hp && hover !== null && (
            <g pointerEvents="none">
              <line x1={x(hover)} x2={x(hover)} y1={pad.top} y2={pad.top + h} stroke="var(--axis)" strokeWidth={1} />
              {hp.value !== null && (
                <circle cx={x(hover)} cy={y(hp.value)} r={4} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />
              )}
            </g>
          )}
        </svg>
      )}
      {hp && hover !== null && (
        <div
          role="status"
          className="pointer-events-none absolute top-1 rounded-md border border-line bg-surface px-2 py-1 text-xs shadow-[var(--shadow)]"
          style={{ left: Math.min(Math.max(x(hover) - 50, 0), Math.max(width - 110, 0)) }}
        >
          <div className="text-muted">{hp.label}</div>
          <div className="tabular font-medium text-ink">{hp.value === null ? "no data" : format(hp.value)}</div>
        </div>
      )}
    </div>
  );
}

/** 12-point trend for a stat tile: de-emphasised line, current point in the accent. */
export function Sparkline({ values }: { values: (number | null)[] }) {
  const pts = values.slice(-12);
  const nums = pts.filter((v): v is number => v !== null);
  if (nums.length < 2) return null;
  const max = Math.max(...nums);
  const min = Math.min(...nums);
  const W = 96;
  const H = 28;
  const x = (i: number) => (i / (pts.length - 1)) * (W - 6) + 3;
  const y = (v: number) => H - 4 - ((v - min) / (max - min || 1)) * (H - 8);
  const d = pts
    .map((v, i) => (v === null ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`))
    .filter(Boolean)
    .join("L");
  const lastV = pts.at(-1);
  return (
    <svg width={W} height={H} aria-hidden className="overflow-visible">
      <path d={`M${d}`} fill="none" stroke="var(--axis)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
      {lastV !== null && lastV !== undefined && (
        <circle cx={x(pts.length - 1)} cy={y(lastV)} r={3} fill="var(--accent)" stroke="var(--surface)" strokeWidth={1.5} />
      )}
    </svg>
  );
}
