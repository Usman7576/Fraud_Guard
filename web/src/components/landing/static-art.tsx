// Static stand-ins for the 3D scenes: shown first, and kept when WebGL is missing or the visitor
// prefers reduced motion. Same palette and the same story, without motion.
import { COLORS } from "./use-3d";

function project(lat: number, lon: number, r: number): [number, number, number] {
  const x = Math.cos(lat) * Math.sin(lon);
  const y = Math.sin(lat);
  const z = Math.cos(lat) * Math.cos(lon);
  return [x * r, -y * r, z];
}

const DOTS = (() => {
  const out: [number, number, number][] = [];
  const n = 700;
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / (n - 1)) * 2;
    const lon = golden * i;
    const p = project(Math.asin(y), lon, 150);
    // Rounded so server and browser trigonometry print the same attributes (hydration).
    if (p[2] > 0) out.push([Math.round(p[0] * 100) / 100, Math.round(p[1] * 100) / 100, Math.round(p[2] * 100) / 100]);
  }
  return out;
})();

const ARCS: { d: string; fraud?: boolean }[] = [
  { d: "M-110,-40 Q-20,-150 90,-70" },
  { d: "M-60,70 Q20,-30 120,20" },
  { d: "M-130,20 Q-80,-60 -10,-100" },
  { d: "M20,110 Q70,30 130,-30" },
  { d: "M-90,-90 Q0,-20 60,90", fraud: true },
  { d: "M-140,-10 Q-60,80 30,120" },
];

export function StaticGlobe() {
  return (
    <svg viewBox="-200 -200 400 400" className="size-full" aria-hidden>
      <defs>
        <radialGradient id="fg-glow" r="0.5">
          <stop offset="0.72" stopColor={COLORS.accent} stopOpacity="0" />
          <stop offset="0.8" stopColor={COLORS.accent} stopOpacity="0.28" />
          <stop offset="1" stopColor={COLORS.accent} stopOpacity="0" />
        </radialGradient>
      </defs>
      <circle r="190" fill="url(#fg-glow)" />
      {DOTS.map(([x, y, z], i) => (
        <circle key={i} cx={x} cy={y} r={1.4} fill={COLORS.accentBright} opacity={0.12 + z * 0.45} />
      ))}
      {ARCS.map((a) => (
        <path
          key={a.d}
          d={a.d}
          fill="none"
          stroke={a.fraud ? COLORS.critical : COLORS.accent}
          strokeWidth={a.fraud ? 2 : 1.5}
          strokeLinecap="round"
          pathLength={1}
          strokeDasharray={a.fraud ? "0.62 1" : undefined}
        />
      ))}
      {/* the fraud arc is stopped by the shield ring */}
      <circle cx={10} cy={12} r={9} fill="none" stroke={COLORS.accentBright} strokeWidth={1.5} />
      <ellipse rx="196" ry="46" fill="none" stroke={COLORS.accentBright} strokeOpacity={0.4} transform="rotate(-14)" />
    </svg>
  );
}

// Entity graph: the final state (two hops lit) of the scroll story.
export const GRAPH_NODES = {
  cards: [
    { id: "A", x: 0, y: 0, ring: "seed" },
    { id: "B", x: 150, y: -70, ring: "hop2" },
    { id: "C", x: 160, y: 60, ring: "hop2" },
    { id: "D", x: -120, y: 120, ring: "hop2" },
    { id: "E", x: -230, y: -110, ring: "" },
    { id: "F", x: 260, y: 150, ring: "" },
    { id: "G", x: -260, y: 60, ring: "" },
  ],
  links: [
    { id: "dev", kind: "device", x: 80, y: -5, ring: "hop1", to: ["A", "B", "C"] },
    { id: "mail", kind: "email", x: -70, y: 50, ring: "hop1", to: ["A", "D"] },
    { id: "dev2", kind: "device", x: -220, y: -20, ring: "", to: ["E", "G"] },
    { id: "mail2", kind: "email", x: 250, y: 50, ring: "", to: ["F"] },
  ],
} as const;

function ringColor(ring: string): string {
  if (ring === "seed") return COLORS.critical;
  if (ring === "hop2") return COLORS.risk;
  if (ring === "hop1") return COLORS.warning;
  return COLORS.node;
}

export function StaticGraph() {
  const cards = Object.fromEntries(GRAPH_NODES.cards.map((c) => [c.id, c]));
  return (
    <svg viewBox="-320 -190 640 380" className="size-full" role="img" aria-label="Card A's chargeback reaches cards B, C and D through a shared device and email">
      {GRAPH_NODES.links.flatMap((l) =>
        l.to.map((id) => {
          const c = cards[id]!;
          return (
            <line
              key={l.id + id}
              x1={l.x}
              y1={l.y}
              x2={c.x}
              y2={c.y}
              stroke={l.ring ? ringColor(c.ring === "seed" ? "hop1" : c.ring) : COLORS.edge}
              strokeOpacity={l.ring ? 0.8 : 1}
              strokeWidth={l.ring ? 2 : 1.5}
            />
          );
        }),
      )}
      {GRAPH_NODES.links.map((l) =>
        l.kind === "device" ? (
          <rect key={l.id} x={l.x - 9} y={l.y - 9} width={18} height={18} transform={`rotate(45 ${l.x} ${l.y})`} fill={ringColor(l.ring)} />
        ) : (
          <circle key={l.id} cx={l.x} cy={l.y} r={10} fill={ringColor(l.ring)} />
        ),
      )}
      {GRAPH_NODES.cards.map((c) => (
        <g key={c.id}>
          <rect x={c.x - 22} y={c.y - 14} width={44} height={28} rx={5} fill={ringColor(c.ring)} />
          <text x={c.x} y={c.y} dy="0.35em" textAnchor="middle" fontSize={12} fontWeight={600} fill={COLORS.background}>
            {c.id}
          </text>
        </g>
      ))}
    </svg>
  );
}
