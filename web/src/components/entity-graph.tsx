// The card's entity neighbourhood: the card in the middle, its devices, addresses and email
// domains around it, each sized by how many cards share it. Entities shared with confirmed
// fraud cards get a red ring and a "fraud" count in words.
import type { Neighbour } from "@/lib/types";

const KIND_LABEL = { device: "Device", address: "Address", email: "Email" } as const;

export function EntityGraph({ card, neighbourhood }: { card: string; neighbourhood: Record<keyof typeof KIND_LABEL, Neighbour[]> }) {
  const nodes = (Object.keys(KIND_LABEL) as (keyof typeof KIND_LABEL)[]).flatMap((kind) =>
    (neighbourhood[kind] ?? []).slice(0, 6).map((n) => ({ kind, ...n })),
  );
  const W = 360;
  const H = 240;
  const cx = W / 2;
  const cy = H / 2;
  const r = 88;

  if (!nodes.length) {
    return <p className="px-5 pb-5 text-sm text-muted">This card has no linked devices or addresses yet.</p>;
  }
  return (
    <div className="px-3 pb-4">
      <svg viewBox={`0 0 ${W} ${H}`} className="mx-auto block w-full max-w-md" role="img" aria-label={`Entities linked to card ${card}`}>
        {nodes.map((n, i) => {
          const a = (i / nodes.length) * 2 * Math.PI - Math.PI / 2;
          const nx = cx + r * Math.cos(a);
          const ny = cy + r * Math.sin(a);
          const size = 6 + Math.min(Math.log2(n.cards + 1) * 2.5, 12);
          const fraud = n.fraud_cards > 0;
          return (
            <g key={`${n.kind}-${n.id}`}>
              <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="var(--line)" strokeWidth={1.5} />
              <circle cx={nx} cy={ny} r={size} fill="var(--surface-2)" stroke={fraud ? "var(--critical)" : "var(--axis)"} strokeWidth={fraud ? 2.5 : 1.5}>
                <title>{`${KIND_LABEL[n.kind]} ${n.id}: ${n.cards} cards, ${n.fraud_cards} with confirmed fraud`}</title>
              </circle>
              <text x={nx} y={ny + size + 11} textAnchor="middle" className="fill-[var(--ink-2)] text-[9px]">
                {KIND_LABEL[n.kind]} · {n.cards} cards
              </text>
              {fraud && (
                <text x={nx} y={ny + size + 21} textAnchor="middle" className="fill-[var(--critical-ink)] text-[9px] font-medium">
                  {n.fraud_cards} fraud
                </text>
              )}
            </g>
          );
        })}
        <circle cx={cx} cy={cy} r={16} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />
        <text x={cx} y={cy} dy="0.32em" textAnchor="middle" className="fill-white text-[9px] font-semibold">
          card
        </text>
      </svg>
      <ul className="mt-2 grid gap-1 px-2 text-xs text-ink-2 sm:grid-cols-2">
        {nodes.map((n) => (
          <li key={`l-${n.kind}-${n.id}`} className="truncate">
            <span className="text-muted">{KIND_LABEL[n.kind]}</span> <span className="font-mono">{n.id}</span>
            {n.fraud_cards > 0 ? <span className="text-critical-ink"> · {n.fraud_cards} fraud</span> : null}
          </li>
        ))}
      </ul>
    </div>
  );
}
