"use client";

// Entity graph: cards, devices and emails. Scroll progress (0..1, from GSAP ScrollTrigger) lights a
// confirmed fraud card, then the device and email it used (one hop), then the cards sharing them
// (two hops), while the camera moves in on the ring.
import { Float, Line, RoundedBox } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

import { Label } from "./label";
import { GRAPH_NODES } from "./static-art";
import { COLORS, DPR } from "./use-3d";

type Ring = "" | "seed" | "hop1" | "hop2";

const smooth = (a: number, b: number, x: number) => THREE.MathUtils.smoothstep(x, a, b);

/** How lit each ring is at scroll progress p. */
export function ringLevel(ring: Ring, p: number): number {
  if (ring === "seed") return smooth(0.12, 0.28, p);
  if (ring === "hop1") return smooth(0.38, 0.54, p);
  if (ring === "hop2") return smooth(0.64, 0.8, p);
  return 0;
}

const TARGET: Record<Ring, THREE.Color> = {
  "": new THREE.Color(COLORS.node),
  seed: new THREE.Color(COLORS.critical),
  hop1: new THREE.Color(COLORS.warning),
  hop2: new THREE.Color(COLORS.risk),
};
const BASE = new THREE.Color(COLORS.node);
const EDGE = new THREE.Color(COLORS.edge);

const pos = (x: number, y: number, i: number) => new THREE.Vector3(x / 95, -y / 95, Math.sin(i * 2.3) * 0.5);

interface Node {
  id: string;
  kind: "card" | "device" | "email";
  ring: Ring;
  p: THREE.Vector3;
}

function useGraph() {
  return useMemo(() => {
    const nodes: Node[] = [
      ...GRAPH_NODES.cards.map((c, i) => ({ id: c.id, kind: "card" as const, ring: c.ring as Ring, p: pos(c.x, c.y, i) })),
      ...GRAPH_NODES.links.map((l, i) => ({ id: l.id, kind: l.kind, ring: l.ring as Ring, p: pos(l.x, l.y, i + 7) })),
    ];
    const at = Object.fromEntries(nodes.map((n) => [n.id, n]));
    const edges = GRAPH_NODES.links.flatMap((l) =>
      l.to.map((id) => {
        const card = at[id]!;
        // An edge lights with its outer end: seed<->hop1 at hop 1, hop1<->hop2 at hop 2.
        const ring: Ring = !l.ring ? "" : card.ring === "seed" ? "hop1" : card.ring;
        return { key: l.id + id, a: at[l.id]!.p, b: card.p, ring };
      }),
    );
    return { nodes, edges };
  }, []);
}

function NodeMesh({ node, progress }: { node: Node; progress: React.RefObject<number> }) {
  const mat = useRef<THREE.MeshStandardMaterial>(null);
  const mesh = useRef<THREE.Group>(null);
  useFrame(({ clock }) => {
    const k = ringLevel(node.ring, progress.current);
    if (mat.current) {
      mat.current.color.copy(BASE).lerp(TARGET[node.ring], k);
      mat.current.emissive.copy(TARGET[node.ring]).multiplyScalar(k * 0.45);
    }
    if (mesh.current) mesh.current.scale.setScalar(1 + k * 0.12 * (1 + Math.sin(clock.elapsedTime * 3) * 0.5));
  });
  const material = <meshStandardMaterial ref={mat} color={COLORS.node} roughness={0.45} metalness={0.1} />;
  return (
    <group ref={mesh} position={node.p}>
      {node.kind === "card" ? (
        <RoundedBox args={[0.46, 0.3, 0.05]} radius={0.04} smoothness={3}>
          {material}
        </RoundedBox>
      ) : node.kind === "device" ? (
        <mesh>
          <octahedronGeometry args={[0.13]} />
          {material}
        </mesh>
      ) : (
        <mesh>
          <sphereGeometry args={[0.11, 24, 24]} />
          {material}
        </mesh>
      )}
      {node.kind === "card" && (
        <Label text={node.id} position={[0, 0, 0.03]} height={0.17} color={COLORS.background} weight={700} />
      )}
    </group>
  );
}

function Edge({ a, b, ring, progress }: { a: THREE.Vector3; b: THREE.Vector3; ring: Ring; progress: React.RefObject<number> }) {
  const line = useRef<React.ComponentRef<typeof Line>>(null);
  useFrame(() => {
    const k = ringLevel(ring, progress.current);
    const m = line.current?.material;
    if (!m) return;
    m.color.copy(EDGE).lerp(TARGET[ring], k);
    m.linewidth = 1.5 + k * 1.5;
  });
  return <Line ref={line} points={[a, b]} color={COLORS.edge} lineWidth={1.5} />;
}

function Rig({ progress }: { progress: React.RefObject<number> }) {
  const { camera } = useThree();
  const look = useMemo(() => new THREE.Vector3(), []);
  const goal = useMemo(() => new THREE.Vector3(), []);
  const target = useMemo(() => new THREE.Vector3(), []);
  useFrame((_, delta) => {
    const p = progress.current;
    const angle = -0.35 + p * 0.6;
    const dist = THREE.MathUtils.lerp(8.8, 6.6, smooth(0.1, 0.85, p));
    goal.set(Math.sin(angle) * dist, 0.4 - p * 0.3, Math.cos(angle) * dist);
    camera.position.lerp(goal, 1 - Math.exp(-delta * 4));
    look.lerp(target.set(0.25 * p, -0.15 * p, 0), 1 - Math.exp(-delta * 4));
    camera.lookAt(look);
  });
  return null;
}

export default function GraphScene({ progress, active }: { progress: React.RefObject<number>; active: boolean }) {
  const { nodes, edges } = useGraph();
  return (
    <Canvas dpr={DPR} frameloop={active ? "always" : "never"} camera={{ position: [0, 0.4, 8.8], fov: 40 }} gl={{ antialias: true }} aria-hidden>
      <color attach="background" args={[COLORS.background]} />
      <ambientLight intensity={0.7} />
      <directionalLight position={[3, 4, 5]} intensity={1.6} />
      <Rig progress={progress} />
      <Float speed={1.1} rotationIntensity={0.15} floatIntensity={0.4}>
        {edges.map((e) => (
          <Edge key={e.key} a={e.a} b={e.b} ring={e.ring} progress={progress} />
        ))}
        {nodes.map((n) => (
          <NodeMesh key={n.id} node={n} progress={progress} />
        ))}
      </Float>
    </Canvas>
  );
}
